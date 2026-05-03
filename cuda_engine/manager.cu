#include "manager.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cub/cub.cuh>

///////////////////////////////////////////////////////////////
///                     Manager Utilities                   ///
///////////////////////////////////////////////////////////////
#define CHECK_CUDA(val) checkCudaError((val), __FILE__, __LINE__)

extern __shared__ float sdata[];


void checkCudaError(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error("CUDA Error at " + std::string(file) + ":" +std::to_string(line) + " - " + cudaGetErrorString(result));
    }
}

int CudaManager::_get_total_elements(const Gpu_Element& element) {
    if (element.sizes.empty()) return 1;
    int total = 1;
    for (int size : element.sizes) {
        total *= size;
    }
    return total;
}

const Gpu_Element& CudaManager::_get_element_no_lock(const std::string& name) {
    return memory.at(name);
}

void CudaManager::_copyHostToDevice_no_lock(const std::string& name, const float* h_ptr) {
    const Gpu_Element& element = _get_element_no_lock(name);
    int total_elements = _get_total_elements(element);
    size_t bytes = total_elements * sizeof(float);
    CHECK_CUDA(cudaMemcpy(element.ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
///                      Lifecycle & Infrastructure                           //
////////////////////////////////////////////////////////////////////////////////

CudaManager::CudaManager() : cublas_handle(nullptr) {
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cublasStatus_t stat = cublasCreate(&this->cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS initialization failed");
    }

    int pool_size = 16;
    stream_pool.resize(pool_size);
    for (int i = 0; i < pool_size; i++) {
        CHECK_CUDA(cudaStreamCreate(&stream_pool[i]));
    }
}

CudaManager::~CudaManager() {
    cublasDestroy(this->cublas_handle);

    // Destroy Stream Pool
    for (auto & i : stream_pool) {
        CHECK_CUDA(cudaStreamDestroy(i));
    }

    // Free all remaining GPU memory
    for (auto it = memory.begin(); it != memory.end(); ++it) {
        cudaFree(it->second.ptr);
    }
    memory.clear();
}

cudaStream_t CudaManager::get_stream_from_pool() const {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    if (stream_pool.empty()) {
        throw std::runtime_error("Stream pool is empty!");
    }
    return stream_pool[0];
}

void CudaManager::synchronize_stream(cudaStream_t stream) const {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

////////////////////////////////////////////////////////////////////////////////
///                          Memory Management                                //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::allocate(const std::string& name, const std::vector<int>& sizes) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    size_t total_elements = 1;
    for (int size : sizes) {
        total_elements *= size;
    }

    Gpu_Element element;
    element.sizes = sizes;

    // Allocate and check for errors during allocation
    CHECK_CUDA(cudaMalloc(&element.ptr, total_elements * sizeof(float)));

    memory[name] = element;
}

bool CudaManager::exists(const std::string& name) const {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    return memory.count(name) > 0;
}

void CudaManager::free(const std::string& name) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    auto it = memory.find(name);
    if (it != memory.end()) {
        CHECK_CUDA(cudaFree(it->second.ptr));
        memory.erase(it);
    }
}

void CudaManager::copyHostToDevice(const std::string& name, const float* h_ptr) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    _copyHostToDevice_no_lock(name, h_ptr);
}

std::vector<float> CudaManager::retrieve(const std::string& name) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& element = _get_element_no_lock(name);

    int total_elements = _get_total_elements(element);
    std::vector<float> h_data(total_elements);
    size_t bytes = total_elements * sizeof(float);

    CHECK_CUDA(cudaMemcpy(h_data.data(), element.ptr, bytes, cudaMemcpyDeviceToHost));
    return h_data;
}

const Gpu_Element& CudaManager::get_element(const std::string& name) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    return _get_element_no_lock(name);
}


////////////////////////////////////////////////////////////////////////////
///                   Basic Math & Element-wise Ops                      //
////////////////////////////////////////////////////////////////////////////

__global__ void copy_kernel(const float* src, float* dest, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dest[i] = src[i];
    }
}

__global__ void fill_kernel(float* data, float value, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        data[i] = value;
    }
}

__global__ void pad2d_kernel(const float* input, float* output, int in_h, int in_w, int out_w, int pad_t, int pad_l, int total_elements) {
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < total_elements) {
        int out_r = idx / out_w;
        int out_c = idx % out_w;

        int in_r = out_r - pad_t;
        int in_c = out_c - pad_l;

        if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
            output[idx] = input[in_r * in_w + in_c];
        } else {
            output[idx] = 0.0f;
        }
    }
}
__global__ void pad2d_backward_kernel(const float* grad_out, float* grad_in, int in_h, int in_w, int out_w, int pad_t, int pad_l, int total_in_elements) {
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < total_in_elements) {
        int in_r = idx / in_w;
        int in_c = idx % in_w;

        int out_r = in_r + pad_t;
        int out_c = in_c + pad_l;

        // Accumulate the gradient from the padded output back into the original input region
        grad_in[idx] = grad_in[idx] + grad_out[out_r * out_w + out_c];
    }
}

__device__ float stateless_random(unsigned int index, unsigned int seed) {
    unsigned int state = index ^ seed;
    // PCG-style avalanche hash
    state ^= state >> 16;
    state *= 0x7feb352du;
    state ^= state >> 15;
    state *= 0x846ca68bu;
    state ^= state >> 16;
    // Convert to float between -1.0 and 1.0
    return (static_cast<float>(state) / 4294967295.0f) * 2.0f - 1.0f;
}

__global__ void init_random_uniform_kernel(float* data, float scale, int seed, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        data[i] = stateless_random(i, seed) * scale;
    }
}

__global__ void add_scalar_kernel(const float* input, float scalar, float* output, int total_elements) {
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < total_elements) {
        output[idx] = input[idx] + scalar;
    }
}


__global__ void element_wise_add_kernel(const float* A, const float* B, float* C, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
__global__ void add_into_kernel(const float* src, float* dest, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dest[i] += src[i];
    }
}
__global__ void add_into_kernel_optimized(const float* src, float* dest, int n) {
    int block_size = static_cast<int>(blockDim.x);
    int grid_size = static_cast<int>(gridDim.x);
    int stride = grid_size * block_size;
    int i = static_cast<int>(blockIdx.x * block_size + threadIdx.x);

    while (i < n) {
        dest[i] += src[i];
        i += stride;
    }
}
__global__ void add_kernel_optimized(const float* A, const float* B, float* C, int n) {
    int stride = static_cast<int>(gridDim.x * blockDim.x);
    int thread_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    // --- Vectorized (float4) Part ---
    const auto* A4 = reinterpret_cast<const float4 *>(A);
    const auto* B4 = reinterpret_cast<const float4 *>(B);
    auto* C4 = reinterpret_cast<float4 *>(C);
    int n4 = n / 4;

    for (int i = thread_idx; i < n4; i += stride) {
        float4 a_vec = A4[i];
        float4 b_vec = B4[i];
        float4 c_vec;
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        C4[i] = c_vec;
    }

    // --- Cleanup Part ---
    int remainder_start = n4 * 4;
    for (int k = remainder_start + thread_idx; k < n; k += stride) {
        C[k] = A[k] + B[k];
    }
}


__global__ void elementwise_subtract_kernel(const float* A, const float* B, float* C, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}
__global__ void subtract_into_kernel(const float* src, float* dest, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dest[i] -= src[i];
    }
}

__global__ void elementwise_multiply_kernel(const float* A, const float* B, float* C, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}
__global__ void elementwise_divide_kernel(const float* A, const float* B, float* C, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        C[i] = A[i] / B[i];
    }
}

__global__ void exp_forward_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = expf(in[i]);
    }
}
__global__ void exp_backward_kernel(const float* grad_out, const float* out_data, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        // dL/din = grad_out * exp(in) = grad_out * out_data
        grad_in[i] += grad_out[i] * out_data[i];
    }
}

__global__ void multiply_backward_kernel(const float* grad_out, const float* other_input, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        grad_in[i] += grad_out[i] * other_input[i];
    }
}
__global__ void divide_backward_kernel(const float* A, const float* B, const float* grad_out, float* grad_A, float* grad_B, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        // dL/dA = dL/dC * (1/B)
        grad_A[i] += grad_out[i] / B[i];
        // dL/dB = dL/dC * (-A / B^2)
        grad_B[i] += grad_out[i] * (-A[i] / (B[i] * B[i]));
    }
}

__global__ void scale_kernel(float* data, float scalar, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        data[i] *= scalar;
    }
}

__global__ void log_forward_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = logf(fmaxf(in[i], 1e-7f));
}
__global__ void log_backward_kernel(const float* grad_out, const float* in_data, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) grad_in[i] += grad_out[i] / fmaxf(in_data[i], 1e-7f);
}

__global__ void abs_forward_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = fabsf(in[i]);
}
__global__ void abs_backward_kernel(const float* grad_out, const float* in_data, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float val = in_data[i];
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        grad_in[i] += grad_out[i] * sign;
    }
}

__global__ void sqrt_forward_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = sqrtf(fmaxf(in[i], 1e-7f));
}
__global__ void sqrt_backward_kernel(const float* grad_out, const float* out_data, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) grad_in[i] += grad_out[i] / (2.0f * fmaxf(out_data[i], 1e-7f));
}

__global__ void pow_forward_kernel(const float* in, float* out, float exponent, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = powf(in[i], exponent);
}
__global__ void pow_backward_kernel(const float* grad_out, const float* in_data, float* grad_in, float exponent, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) grad_in[i] += grad_out[i] * exponent * powf(in_data[i], exponent - 1.0f);
}

__global__ void clamp_forward_kernel(const float* in, float* out, float min_val, float max_val, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = fminf(max_val, fmaxf(min_val, in[i]));
}
__global__ void clamp_backward_kernel(const float* grad_out, const float* in_data, float* grad_in, float min_val, float max_val, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float val = in_data[i];
        if (val >= min_val && val <= max_val) {
            grad_in[i] += grad_out[i];
        }
    }
}


///////////////////////////////////////////////////////
///                 Matrix Operations               ///
///////////////////////////////////////////////////////
#define TILE_DIM 32
__global__ void transpose_shared_mem_kernel(const float* in, float* out, int width, int height) {
    // +1 padding to avoid shared memory bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = static_cast<int>(blockIdx.x * TILE_DIM + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * TILE_DIM + threadIdx.y);

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    x = static_cast<int>(blockIdx.y * TILE_DIM + threadIdx.x);
    y = static_cast<int>(blockIdx.x * TILE_DIM + threadIdx.y);

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void broadcast_add_kernel(const float* A, const float* B, float* C, int M, int N, int b_type) {
    int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (row < M && col < N) {
        float b_val = 0;
        if (b_type == 0) b_val = B[0];
        else if (b_type == 1) b_val = B[col];
        else if (b_type == 2) b_val = B[row];

        C[row * N + col] = A[row * N + col] + b_val;
    }
}



////////////////////////////////////////////////////////////////////////////////
///                          Activation Functions                             //
////////////////////////////////////////////////////////////////////////////////

__global__ void relu_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = fmaxf(0.0f, in[i]);
    }
}
__global__ void relu_backward_kernel(const float* in, const float* grad_out, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        grad_in[i] += grad_out[i] * (in[i] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void tanh_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = tanhf(in[i]);
    }
}
__global__ void tanh_backward_kernel(const float* out_data, const float* grad_out, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float tanh_val = out_data[i];
        grad_in[i] += grad_out[i] * (1.0f - tanh_val * tanh_val);
    }
}

__global__ void sigmoid_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}
__global__ void sigmoid_backward_kernel(const float* out_data, const float* grad_out, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float sigmoid_val = out_data[i];
        grad_in[i] += grad_out[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
}

__global__ void gelu_forward_kernel(const float* in, float* out, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float x = in[i];
        // 0.707106781f is 1 / sqrt(2)
        out[i] = 0.5f * x * (1.0f + erff(x * 0.707106781f));
    }
}
__global__ void gelu_backward_kernel(const float* in, const float* grad_out, float* grad_in, int n) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float x = in[i];
        float cdf = 0.5f * (1.0f + erff(x * 0.707106781f));
        // 0.3989422804f is 1 / sqrt(2 * pi)
        float pdf = 0.3989422804f * expf(-0.5f * x * x);

        grad_in[i] += grad_out[i] * (cdf + x * pdf);
    }
}

__global__ void softmax_forward_kernel(const float* in, float* out, int M_rows, int N_cols) {
    int row = static_cast<int>(blockIdx.x);
    if (row >= M_rows) return;

    int tid = static_cast<int>(threadIdx.x);
    const float* row_in = in + row * N_cols;
    float* row_out = out + row * N_cols;

    // 1. Thread-local Max
    float local_max = -__FLT_MAX__;
    for (int i = tid; i < N_cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Block-wide Max Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // 2. Thread-local Sum of Exps
    float local_sum = 0.0f;
    for (int i = tid; i < N_cols; i += blockDim.x) {
        local_sum += expf(row_in[i] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block-wide Sum Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exps = sdata[0];
    __syncthreads();

    // 3. Normalize (Recompute exp to save shared memory)
    for (int i = tid; i < N_cols; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_val) / sum_exps;
    }
}
__global__ void softmax_backward_kernel(const float* grad_out, const float* out_data, float* grad_in, int M_rows, int N_cols) {
    int row = static_cast<int>(blockIdx.x);
    if (row >= M_rows) return;

    int tid = static_cast<int>(threadIdx.x);
    const float* row_grad_out = grad_out + row * N_cols;
    const float* row_out_data = out_data + row * N_cols;
    float* row_grad_in = grad_in + row * N_cols;

    // 1. Thread-local Dot Product
    float local_dot = 0.0f;
    for (int i = tid; i < N_cols; i += blockDim.x) {
        local_dot += row_grad_out[i] * row_out_data[i];
    }
    sdata[tid] = local_dot;
    __syncthreads();

    // Block-wide Dot Product Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float dot_prod = sdata[0];
    __syncthreads();

    // 2. Compute and Accumulate Gradient
    for (int i = tid; i < N_cols; i += blockDim.x) {
        row_grad_in[i] += row_out_data[i] * (row_grad_out[i] - dot_prod);
    }
}


////////////////////////////////////////////////////////////////////////////////
///                             Loss Functions                                //
////////////////////////////////////////////////////////////////////////////////

__global__ void mse_forward_kernel(const float* pred, const float* target, float* out, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (pred[i] - target[i]) * (pred[i] - target[i]) : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = sdata[0] / n; // Note: Simple single-block reduction
    }
}
__global__ void mse_backward_kernel(const float* pred, const float* target, float* grad_out, const float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad_in[0];
        grad_out[i] += g * 2.0f * (pred[i] - target[i]) / static_cast<float>(n);
    }
}

__global__ void elementwise_squared_error_kernel(const float* pred, const float* target, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float error = pred[i] - target[i];
        out[i] = error * error;
    }
}

__global__ void bce_elementwise_kernel(const float* pred, const float* target, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float p = fminf(0.999999f, fmaxf(1e-9f, pred[i]));
        out[i] = - (target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p));
    }
}
__global__ void bce_backward_kernel(const float* grad_out, const float* pred, const float* target, float* grad_in, int n) {
    float g = grad_out[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float p = fminf(0.999999f, fmaxf(1e-9f, pred[i]));
        grad_in[i] += g * ((p - target[i]) / (p * (1.0f - p)));
    }
}

////////////////////////////////////////////////////////////////////////////////
///                               Optimizers                                  //
////////////////////////////////////////////////////////////////////////////////

__global__ void sgd_update_kernel(float* data, const float* grad, float learning_rate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] - (learning_rate * grad[i]);
    }
}

__global__ void adam_update_kernel(float* data, const float* grad, float* m, float* v,float lr, float beta1, float beta2, float eps,float one_minus_b1t, float one_minus_b2t, float weight_decay, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float mi = m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);
        float m_hat = mi / one_minus_b1t;
        float v_hat = vi / one_minus_b2t;
        data[i] -= lr * (m_hat / (sqrtf(v_hat) + eps));
        if (weight_decay != 0.0f) {
            data[i] -= lr * weight_decay * data[i];
        }
    }
}

__global__ void clip_grad_value_kernel(float* data, float clip_value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        float abs_clip_val = fabsf(clip_value);
        data[i] = fminf(abs_clip_val, fmaxf(-abs_clip_val, val));
    }
}

////////////////////////////////////////////////////////////////////////////////
///                    Advanced / Fused Layers                                //
////////////////////////////////////////////////////////////////////////////////

__global__ void add_bias_relu_store_pre_relu_kernel(const float* matmul_result, const float* bias, float* relu_output,float* pre_relu_output, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        float intermediate_val = matmul_result[idx];

        intermediate_val += bias[col];

        pre_relu_output[idx] = intermediate_val;
        relu_output[idx] = fmaxf(0.0f, intermediate_val);
    }
}

__global__ void layer_norm_forward_kernel(const float* in, const float* gamma, const float* beta,float* out, float* mean_cache, float* rstd_cache,int M_rows, int N_cols, float epsilon)
{
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int tid = threadIdx.x;
    const float* row_in = in + row * N_cols;
    float* row_out = out + row * N_cols;

    // 1. Calculate Mean using grid-stride
    float local_sum = 0.0f;
    for (int i = tid; i < N_cols; i += blockDim.x) {
        local_sum += row_in[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / N_cols;
    __syncthreads();

    // 2. Calculate Variance using grid-stride
    float local_var = 0.0f;
    for (int i = tid; i < N_cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / N_cols;

    float rstd = rsqrtf(variance + epsilon);
    if (tid == 0) {
        mean_cache[row] = mean;
        rstd_cache[row] = rstd;
    }
    __syncthreads(); // Ensure everyone waits before applying the final pass

    // 3. Apply Normalization
    for (int i = tid; i < N_cols; i += blockDim.x) {
        float x_hat = (row_in[i] - mean) * rstd;
        row_out[i] = x_hat * gamma[i] + beta[i];
    }
}
__global__ void layer_norm_backward_kernel(const float* grad_out, const float* in, const float* gamma,const float* mean_cache, const float* rstd_cache,float* grad_in, float* grad_gamma, float* grad_beta,int M_rows, int N_cols)
{
    int row = blockIdx.x;
    if (row >= M_rows) return;

    int tid = threadIdx.x;

    const float* row_grad_out = grad_out + row * N_cols;
    const float* row_in = in + row * N_cols;
    float* row_grad_in = grad_in + row * N_cols;

    float mean = mean_cache[row];
    float rstd = rstd_cache[row];

    float local_sum_grad_x_hat = 0.0f;
    float local_sum_grad_x_hat_x_hat = 0.0f;

    // 1. First pass: Accumulate thread-local sums and apply atomics
    for (int i = tid; i < N_cols; i += blockDim.x) {
        float x_hat = (row_in[i] - mean) * rstd;
        float grad_x_hat = row_grad_out[i] * gamma[i];

        local_sum_grad_x_hat += grad_x_hat;
        local_sum_grad_x_hat_x_hat += grad_x_hat * x_hat;

        atomicAdd(&grad_beta[i], row_grad_out[i]);
        atomicAdd(&grad_gamma[i], row_grad_out[i] * x_hat);
    }

    // 2. Reduce the sums across the block
    sdata[tid] = local_sum_grad_x_hat;
    sdata[tid + blockDim.x] = local_sum_grad_x_hat_x_hat;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }

    float sum_grad_x_hat = sdata[0];
    float sum_grad_x_hat_x_hat = sdata[blockDim.x];
    __syncthreads();

    // 3. Final pass: Recompute x_hat on the fly and compute gradient
    auto N_f = static_cast<float>(N_cols);
    for (int i = tid; i < N_cols; i += blockDim.x) {
        float x_hat = (row_in[i] - mean) * rstd;           // Recomputed
        float grad_x_hat = row_grad_out[i] * gamma[i];     // Recomputed

        float term1 = N_f * grad_x_hat;
        float term2 = sum_grad_x_hat;
        float term3 = x_hat * sum_grad_x_hat_x_hat;

        row_grad_in[i] += (rstd / N_f) * (term1 - term2 - term3);
    }
}

__global__ void conv2d_forward_kernel(const float* input, const float* kernel, float* output,int inHeight, int inWidth, int kernelHeight, int kernelWidth,int outHeight, int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int in_y = y + ky;
                int in_x = x + kx;
                sum += input[in_y * inWidth + in_x] * kernel[ky * kernelWidth + kx];
            }
        }
        output[y * outWidth + x] = sum;
    }
}
__global__ void conv2d_backward_input_kernel(const float* grad_out, const float* kernel, float* grad_in,int inHeight, int inWidth, int kernelHeight, int kernelWidth,int outHeight, int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        float grad = grad_out[y * outWidth + x];
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int in_y = y + ky;
                int in_x = x + kx;
                atomicAdd(&grad_in[in_y * inWidth + in_x], kernel[ky * kernelWidth + kx] * grad);
            }
        }
    }
}
__global__ void conv2d_backward_kernel_kernel(const float* input, const float* grad_out, float* grad_kernel,int inHeight, int inWidth, int kernelHeight, int kernelWidth,int outHeight, int outWidth)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;

    if (kx < kernelWidth && ky < kernelHeight) {
        float sum = 0.0f;
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                int in_y = y + ky;
                int in_x = x + kx;
                sum += input[in_y * inWidth + in_x] * grad_out[y * outWidth + x];
            }
        }
        atomicAdd(&grad_kernel[ky * kernelWidth + kx], sum);
    }
}

__global__ void conv2d_multi_forward_kernel(const float* input, const float* weight, const float* bias, float* output,int C_in, int in_h, int in_w, int C_out, int k_h, int k_w, int out_h, int out_w, int pad_t, int pad_l, int stride_h, int stride_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x < out_w && out_y < out_h && out_c < C_out) {
        float sum = bias[out_c];
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int ky = 0; ky < k_h; ++ky) {
                for (int kx = 0; kx < k_w; ++kx) {
                    // Applied stride multipliers here
                    int in_y = (out_y * stride_h) + ky - pad_t;
                    int in_x = (out_x * stride_w) + kx - pad_l;

                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        int in_idx = c_in * (in_h * in_w) + in_y * in_w + in_x;
                        int w_idx = out_c * (C_in * k_h * k_w) + c_in * (k_h * k_w) + ky * k_w + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        int out_idx = out_c * (out_h * out_w) + out_y * out_w + out_x;
        output[out_idx] = sum;
    }
}
__global__ void conv2d_multi_backward_input_kernel(const float* grad_out, const float* weight, float* grad_in,int C_in, int in_h, int in_w, int C_out, int k_h, int k_w, int out_h, int out_w, int pad_t, int pad_l, int stride_h, int stride_w)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int in_c = blockIdx.z * blockDim.z + threadIdx.z;

    if (in_x < in_w && in_y < in_h && in_c < C_in) {
        float sum = 0.0f;
        for (int out_c = 0; out_c < C_out; ++out_c) {
            for (int ky = 0; ky < k_h; ++ky) {
                for (int kx = 0; kx < k_w; ++kx) {
                    int out_y_raw = in_y - ky + pad_t;
                    int out_x_raw = in_x - kx + pad_l;

                    // Modulo logic to skip gradients that fall in the "gaps" of the stride
                    if (out_y_raw % stride_h == 0 && out_x_raw % stride_w == 0) {
                        int out_y = out_y_raw / stride_h;
                        int out_x = out_x_raw / stride_w;

                        if (out_y >= 0 && out_y < out_h && out_x >= 0 && out_x < out_w) {
                            int grad_out_idx = out_c * (out_h * out_w) + out_y * out_w + out_x;
                            int w_idx = out_c * (C_in * k_h * k_w) + in_c * (k_h * k_w) + ky * k_w + kx;
                            sum += grad_out[grad_out_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        int in_idx = in_c * (in_h * in_w) + in_y * in_w + in_x;
        grad_in[in_idx] += sum;
    }
}
__global__ void conv2d_multi_backward_weight_kernel(const float* input, const float* grad_out, float* grad_weight,int C_in, int in_h, int in_w, int C_out, int k_h, int k_w, int out_h, int out_w, int pad_t, int pad_l, int stride_h, int stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = C_out * C_in * k_h * k_w;
    if (idx < total_weights) {
        int kx = idx % k_w;
        int ky = (idx / k_w) % k_h;
        int in_c = (idx / (k_w * k_h)) % C_in;
        int out_c = idx / (k_w * k_h * C_in);

        float sum = 0.0f;
        for (int out_y = 0; out_y < out_h; ++out_y) {
            for (int out_x = 0; out_x < out_w; ++out_x) {
                // Applied stride multipliers here
                int in_y = (out_y * stride_h) + ky - pad_t;
                int in_x = (out_x * stride_w) + kx - pad_l;

                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    int in_idx = in_c * (in_h * in_w) + in_y * in_w + in_x;
                    int grad_out_idx = out_c * (out_h * out_w) + out_y * out_w + out_x;
                    sum += input[in_idx] * grad_out[grad_out_idx];
                }
            }
        }
        grad_weight[idx] += sum;
    }
}
__global__ void conv2d_multi_backward_bias_kernel(const float* grad_out, float* grad_bias, int C_out, int out_h, int out_w) {
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_c < C_out) {
        float sum = 0.0f;
        int area = out_h * out_w;
        for (int i = 0; i < area; ++i) {
            sum += grad_out[out_c * area + i];
        }
        grad_bias[out_c] += sum;
    }
}

__global__ void im2col_kernel(const float* input, float* col_buffer,int inHeight, int inWidth, int kernelHeight, int kernelWidth,int outHeight, int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        int col_buffer_col_idx = y * outWidth + x;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int in_y = y + ky;
                int in_x = x + kx;
                int col_buffer_row_idx = ky * kernelWidth + kx;
                int col_buffer_idx = col_buffer_row_idx * (outHeight * outWidth) + col_buffer_col_idx;

                col_buffer[col_buffer_idx] = input[in_y * inWidth + in_x];
            }
        }
    }
}
__global__ void col2im_kernel(const float* col_buffer_grad, float* grad_in,int inHeight, int inWidth, int kernelHeight, int kernelWidth,int outHeight, int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        int col_buffer_col_idx = y * outWidth + x;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int in_y = y + ky;
                int in_x = x + kx;
                int col_buffer_row_idx = ky * kernelWidth + kx;
                int col_buffer_idx = col_buffer_row_idx * (outHeight * outWidth) + col_buffer_col_idx;

                float grad_val = col_buffer_grad[col_buffer_idx];
                atomicAdd(&grad_in[in_y * inWidth + in_x], grad_val);
            }
        }
    }
}

__global__ void max_pool_1d_forward_kernel(const float* in, float* out, float* indices,int in_len, int out_len, int pool_size, int stride)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_len) {
        int start_idx = out_idx * stride;
        float max_val = -__FLT_MAX__;
        int max_idx = -1;

        for (int p = 0; p < pool_size; p++) {
            int in_idx = start_idx + p;
            if (in_idx < in_len) {
                float val = in[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }
        out[out_idx] = max_val;
        indices[out_idx] = static_cast<float>(max_idx); // Store flattened index as float
    }
}
__global__ void max_pool_1d_backward_kernel(const float* grad_out, const float* indices, float* grad_in, int out_len)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_len) {
        int max_idx = static_cast<int>(indices[out_idx]);
        if (max_idx >= 0) {
            atomicAdd(&grad_in[max_idx], grad_out[out_idx]);
        }
    }
}

__global__ void max_pool_2d_forward_kernel(const float* in, float* out, float* indices,int in_h, int in_w, int out_h, int out_w, int pool_size, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_w && out_y < out_h) {
        int out_idx = out_y * out_w + out_x;
        int start_y = out_y * stride;
        int start_x = out_x * stride;

        float max_val = -__FLT_MAX__;
        int max_idx = -1;

        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int in_y = start_y + py;
                int in_x = start_x + px;
                if (in_y < in_h && in_x < in_w) {
                    int in_idx = in_y * in_w + in_x;
                    float val = in[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = in_idx; // Store flattened index as float
                    }
                }
            }
        }
        out[out_idx] = max_val;
        indices[out_idx] = static_cast<float>(max_idx);
    }
}
__global__ void max_pool_2d_backward_kernel(const float* grad_out, const float* indices, float* grad_in, int out_total)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_total) {
        int max_idx = static_cast<int>(indices[out_idx]);
        if (max_idx >= 0) {
            atomicAdd(&grad_in[max_idx], grad_out[out_idx]);
        }
    }
}

__global__ void avg_pool_2d_forward_kernel(const float* in, float* out,int in_h, int in_w, int out_h, int out_w, int pool_size, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_w && out_y < out_h) {
        int out_idx = out_y * out_w + out_x;
        int start_y = out_y * stride;
        int start_x = out_x * stride;

        float sum = 0.0f;
        auto num_elements = static_cast<float>(pool_size * pool_size);

        for (int py = 0; py < pool_size; py = py + 1) {
            for (int px = 0; px < pool_size; px = px + 1) {
                int in_y = start_y + py;
                int in_x = start_x + px;
                if (in_y < in_h && in_x < in_w) {
                    sum = sum + in[in_y * in_w + in_x];
                }
            }
        }
        out[out_idx] = sum / num_elements;
    }
}
__global__ void avg_pool_2d_backward_kernel(const float* grad_out, float* grad_in,int in_h, int in_w, int out_h, int out_w, int pool_size, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_w && out_y < out_h) {
        int out_idx = out_y * out_w + out_x;
        auto num_elements = static_cast<float>(pool_size * pool_size);
        float g = grad_out[out_idx] / num_elements;

        int start_y = out_y * stride;
        int start_x = out_x * stride;

        for (int py = 0; py < pool_size; py = py + 1) {
            for (int px = 0; px < pool_size; px = px + 1) {
                int in_y = start_y + py;
                int in_x = start_x + px;
                if (in_y < in_h && in_x < in_w) {
                    atomicAdd(&grad_in[in_y * in_w + in_x], g);
                }
            }
        }
    }
}

__global__ void global_avg_pool_forward_kernel(const float* in, float* out, int seq_len, int d_model) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < d_model) {
        float sum = 0.0f;
        for (int r = 0; r < seq_len; r = r + 1) {
            sum = sum + in[r * d_model + col];
        }
        out[col] = sum / static_cast<float>(seq_len);
    }
}
__global__ void global_avg_pool_backward_kernel(const float* grad_out, float* grad_in, int seq_len, int d_model) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < d_model && row < seq_len) {
        float dist_grad = grad_out[col] / static_cast<float>(seq_len);
        atomicAdd(&grad_in[row * d_model + col], dist_grad);
    }
}


__global__ void batch_norm_1d_forward_kernel(const float* in, const float* gamma, const float* beta,float* running_mean, float* running_var,float* out, float* saved_mean, float* saved_inv_var,int num_features, float momentum, float epsilon, bool is_training)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_features) {
        float current_mean = in[i];
        float current_var = 0.0f; // As per CPU implementation for 1D single-batch

        if (is_training) {
            running_mean[i] = momentum * running_mean[i] + (1.0f - momentum) * current_mean;
            running_var[i] = momentum * running_var[i] + (1.0f - momentum) * current_var;
        }

        // Mirrors dart: meanToUse = isTraining ? runningMean[i] : currentMean[i];
        float mean_to_use = is_training ? running_mean[i] : current_mean;
        // Mirrors dart: varianceToUse = isTraining ? runningVariance : currentVariance;
        float var_to_use = is_training ? running_var[i] : current_var;

        float inv_std = rsqrtf(var_to_use + epsilon);

        saved_mean[i] = mean_to_use;
        saved_inv_var[i] = inv_std;

        float x_hat = (in[i] - mean_to_use) * inv_std;
        out[i] = gamma[i] * x_hat + beta[i];
    }
}
__global__ void batch_norm_1d_backward_kernel(const float* grad_out, const float* in, const float* gamma,const float* saved_mean, const float* saved_inv_var,float* grad_in, float* grad_gamma, float* grad_beta,int num_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_features) {
        float inv_std = saved_inv_var[i];
        float x_hat = (in[i] - saved_mean[i]) * inv_std;

        atomicAdd(&grad_gamma[i], grad_out[i] * x_hat);
        atomicAdd(&grad_beta[i], grad_out[i]);
        atomicAdd(&grad_in[i], grad_out[i] * gamma[i] * inv_std);
    }
}

__global__ void batch_norm_2d_forward_kernel(const float* in, const float* gamma, const float* beta,float* running_mean, float* running_var,float* out, float* saved_mean, float* saved_inv_var,int num_channels, int elements_per_channel,float momentum, float epsilon, bool is_training)
{
    int c = blockIdx.x;
    if (c >= num_channels) return;

    int tid = threadIdx.x;
    const float* channel_in = in + c * elements_per_channel;
    float* channel_out = out + c * elements_per_channel;

    // 1. Compute Mean
    float sum = 0.0f;
    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        sum += channel_in[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float current_mean = sdata[0] / elements_per_channel;
    __syncthreads();

    // 2. Compute Variance
    float var_sum = 0.0f;
    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        float diff = channel_in[i] - current_mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float current_var = sdata[0] / elements_per_channel;
    __syncthreads();

    // 3. Update running stats (Thread 0 only)
    if (tid == 0) {
        if (is_training) {
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * current_mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * current_var;
        }
        float mean_to_use = is_training ? current_mean : running_mean[c];
        float var_to_use = is_training ? current_var : running_var[c];

        saved_mean[c] = mean_to_use;
        saved_inv_var[c] = rsqrtf(var_to_use + epsilon);
    }
    __syncthreads();

    // 4. Normalize
    float mean_to_use = saved_mean[c];
    float inv_std = saved_inv_var[c];
    float g = gamma[c];
    float b = beta[c];

    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        float x_hat = (channel_in[i] - mean_to_use) * inv_std;
        channel_out[i] = g * x_hat + b;
    }
}
__global__ void batch_norm_2d_backward_kernel(const float* grad_out, const float* in, const float* gamma,const float* saved_mean, const float* saved_inv_var,float* grad_in, float* grad_gamma, float* grad_beta,int num_channels, int elements_per_channel)
{
    int c = blockIdx.x;
    if (c >= num_channels) return;

    int tid = threadIdx.x;

    const float* channel_grad_out = grad_out + c * elements_per_channel;
    const float* channel_in = in + c * elements_per_channel;
    float* channel_grad_in = grad_in + c * elements_per_channel;

    float mean = saved_mean[c];
    float inv_std = saved_inv_var[c];
    float g = gamma[c];

    float sum_grad_gamma = 0.0f;
    float sum_grad_beta = 0.0f;

    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        float go = channel_grad_out[i];
        float x_hat = (channel_in[i] - mean) * inv_std;

        sum_grad_gamma += go * x_hat;
        sum_grad_beta += go;

        // Simplified backwards propagation exactly mirroring the CPU fallback
        atomicAdd(&channel_grad_in[i], go * g * inv_std);
    }

    float* s_gamma = sdata;
    float* s_beta = sdata + blockDim.x;

    s_gamma[tid] = sum_grad_gamma;
    s_beta[tid] = sum_grad_beta;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_gamma[tid] += s_gamma[tid + s];
            s_beta[tid] += s_beta[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&grad_gamma[c], s_gamma[0]);
        atomicAdd(&grad_beta[c], s_beta[0]);
    }
}

__global__ void scale_matrix_forward_kernel(const float* in, float* out, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scale;
    }
}
__global__ void scale_matrix_backward_kernel(const float* grad_out, float* grad_in, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] += grad_out[idx] * scale;
    }
}

__device__ float hash_random(unsigned int seed, unsigned int idx) {
    unsigned int state = seed + idx * 0x853c49e6;
    state ^= state >> 13;
    state *= 0x0815451;
    state ^= state >> 11;
    return static_cast<float>(state & 0x00ffffff) / static_cast<float>(0x01000000);
}

__global__ void dropout_forward_kernel(const float* in, float* out, float* mask, float drop_rate, float scale, unsigned int seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float rand_val = hash_random(seed, i);
        if (rand_val < drop_rate) {
            out[i] = 0.0f;
            mask[i] = 0.0f;
        } else {
            out[i] = in[i] * scale;
            mask[i] = scale;
        }
    }
}
__global__ void dropout_backward_kernel(const float* grad_out, const float* mask, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Multiply incoming gradient by the exact mask used during forward pass
        grad_in[i] += grad_out[i] * mask[i];
    }
}

__device__ int history_to_index(const float* history_start, int order) {
    int index = 0;
    for (int i = 0; i < order; ++i) {
        index = (index * 2) + static_cast<int>(history_start[i]);
    }
    return index;
}

__global__ void markov_count_kernel(const float* sequence, int seq_len, float* count_table, int order, int num_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len - order) {
        int history_idx = history_to_index(sequence + i, order);
        int next_state = static_cast<int>(sequence[i + order]);
        int table_idx = history_idx * num_states + next_state;
        atomicAdd(&count_table[table_idx], 1.0f);
    }
}

__global__ void markov_normalize_kernel(const float* count_table, float* prob_table, int num_histories, int num_states) {
    int hist_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hist_idx < num_histories) {
        float sum = 0.0f;
        for (int s = 0; s < num_states; ++s) {
            sum += count_table[hist_idx * num_states + s];
        }
        for (int s = 0; s < num_states; ++s) {
            int idx = hist_idx * num_states + s;
            if (sum > 0.0f) {
                prob_table[idx] = count_table[idx] / sum;
            } else {
                prob_table[idx] = 1.0f / num_states; // Uniform probability if unseen
            }
        }
    }
}

__global__ void markov_predict_kernel(const float* history, const float* prob_table, float* out_probs, int batch_size, int order, int num_states) {
    // Thread represents a single specific output float
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_states;

    if (i < total_elements) {
        int b = i / num_states;    // Which history in the batch?
        int s = i % num_states;    // Which state probability?

        int hist_idx = history_to_index(history + b * order, order);

        // Coalesced write! Thread 0 writes to 0, Thread 1 writes to 1.
        out_probs[i] = prob_table[hist_idx * num_states + s];
    }
}
//////////////////////////////////////////////////////////////////////////////////
///                             Tensor Manipulation                            ///
//////////////////////////////////////////////////////////////////////////////////


__global__ void slice_row_kernel(const float* in, float* out, int in_cols, int row_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_cols) {
        int in_index = row_index * in_cols + i;
        out[i] = in[in_index];
    }
}
__global__ void slice_row_backward_kernel(const float* grad_out, float* grad_in, int in_cols, int row_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_cols) {
        int in_index = row_index * in_cols + i;
        atomicAdd(&grad_in[in_index], grad_out[i]);
    }
}

__global__ void slice_column_kernel(const float* in, float* out, int in_rows, int in_cols_total, int out_cols, int start_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < in_rows && col < out_cols) {
        int in_idx = row * in_cols_total + (col + start_col);
        int out_idx = row * out_cols + col;
        out[out_idx] = in[in_idx];
    }
}
__global__ void slice_column_backward_kernel(const float* grad_out, float* grad_in, int in_rows, int in_cols_total, int out_cols, int start_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < in_rows && col < out_cols) {
        int in_idx = row * in_cols_total + (col + start_col);
        int out_idx = row * out_cols + col;
        atomicAdd(&grad_in[in_idx], grad_out[out_idx]);
    }
}

__global__ void concatenate_kernel(const float* A, const float* B, float* C, int Rows, int Cols_A, int Cols_B, int Cols_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Rows && col < Cols_C) {
        int out_idx = row * Cols_C + col;
        if (col < Cols_A) {
            int in_idx = row * Cols_A + col;
            C[out_idx] = A[in_idx];
        } else {
            int col_in_B = col - Cols_A;
            int in_idx = row * Cols_B + col_in_B;
            C[out_idx] = B[in_idx];
        }
    }
}
__global__ void concatenate_backward_kernel(const float* grad_out, float* grad_in_A, float* grad_in_B, int Rows, int Cols_A, int Cols_B, int Cols_C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Rows && col < Cols_C) {
        int out_idx = row * Cols_C + col;
        float grad_val = grad_out[out_idx];

        if (col < Cols_A) {
            int in_idx = row * Cols_A + col;
            atomicAdd(&grad_in_A[in_idx], grad_val);
        } else {
            int col_in_B = col - Cols_A;
            int in_idx = row * Cols_B + col_in_B;
            atomicAdd(&grad_in_B[in_idx], grad_val);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
///                            Reduction Operations                           //
////////////////////////////////////////////////////////////////////////////////

__global__ void reduce_sum_kernel(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    int gridSize = blockDim.x * 2 * gridDim.x;

    float my_sum = 0;
    while (i < n) {
        my_sum += in[i] + in[i + blockDim.x];
        i += gridSize;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
__global__ void sum_reduce_backward_kernel(const float* grad_out, float* grad_in, int n) {
    float g = grad_out[0];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] += g;
    }
}
__global__ void sum_reduce_columns_kernel(const float* in, float* out, int rows, int cols, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    while (i < n) {
        // Calculate which column this element belongs to
        int col = i % cols;

        // Accumulate into that column's bucket
        atomicAdd(&out[col], in[i]);

        i += stride;
    }
}
__global__ void sum_reduce_rows_kernel(const float* in, float* out, int rows, int cols, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    while (i < n) {
        int row = i / cols;
        atomicAdd(&out[row], in[i]);
        i += stride;
    }
}

__global__ void embedding_forward_kernel(const float* indices, const float* weight, float* out,int num_indices, int embedding_dim, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_indices * embedding_dim;

    if (idx < total_threads) {
        int index_pos = idx / embedding_dim;
        int dim_pos = idx % embedding_dim;

        int word_idx = static_cast<int>(indices[index_pos]);

        if (word_idx >= 0 && word_idx < vocab_size) {
            out[idx] = weight[word_idx * embedding_dim + dim_pos];
        } else {
            out[idx] = 0.0f;
        }
    }
}
__global__ void embedding_backward_kernel(const float* grad_out, const float* indices, float* grad_weight,int num_indices, int embedding_dim, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_indices * embedding_dim;

    if (idx < total_threads) {
        int index_pos = idx / embedding_dim;
        int dim_pos = idx % embedding_dim;

        int word_idx = static_cast<int>(indices[index_pos]);

        if (word_idx >= 0 && word_idx < vocab_size) {
            int weight_idx = word_idx * embedding_dim + dim_pos;
            atomicAdd(&grad_weight[weight_idx], grad_out[idx]);
        }
    }
}


///////////////////////////////////////////////////////////////
/////*****************************************************/////
/////*                CUDA MANAGER METHODS               */////
/////*****************************************************/////
///////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///                    Basic Math & Element-wise Ops                          //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::zero_grad(const std::string& name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& elem = _get_element_no_lock(name);
    int n = _get_total_elements(elem);

    // cudaMemsetAsync is fully optimized, no custom kernel needed
    CHECK_CUDA(cudaMemsetAsync(elem.ptr, 0, n * sizeof(float), stream));
}

void CudaManager::fill(const std::string& name, float value, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& element = _get_element_no_lock(name);
    int total_elements = _get_total_elements(element);
    int blockSize = std::min(256, total_elements);
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    fill_kernel<<<numBlocks, blockSize, 0, stream>>>(element.ptr, value, total_elements);
}

void CudaManager::copy(const std::string& name_src, const std::string& name_dest, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& src = _get_element_no_lock(name_src);
    int n = _get_total_elements(src);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    copy_kernel<<<numBlocks, blockSize, 0, stream>>>(src.ptr, _get_element_no_lock(name_dest).ptr, n);
}

void CudaManager::pad2d(const std::string& name_in, const std::string& name_out, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int in_h = in_elem.sizes[0];
    int in_w = in_elem.sizes[1];
    int out_w = out_elem.sizes[1];
    int total = _get_total_elements(out_elem);

    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    pad2d_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, in_h, in_w, out_w, pad_t, pad_l, total
    );
}
void CudaManager::pad2d_backward(const std::string& name_grad_out, const std::string& name_grad_in, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int in_h = grad_in_elem.sizes[0];
    int in_w = grad_in_elem.sizes[1];
    int out_w = grad_out_elem.sizes[1];
    int total_in = _get_total_elements(grad_in_elem);

    int blockSize = 256;
    int numBlocks = (total_in + blockSize - 1) / blockSize;

    pad2d_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_out_elem.ptr, grad_in_elem.ptr, in_h, in_w, out_w, pad_t, pad_l, total_in
    );
}


void CudaManager::init_random_uniform(const std::string& name, float scale, int seed) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& elem = _get_element_no_lock(name);

    int n = _get_total_elements(elem);
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    init_random_uniform_kernel<<<numBlocks, blockSize>>>(elem.ptr, scale, seed, n);

    // Block the CPU until the GPU finishes writing the random values
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CudaManager::add_scalar(const std::string& name_in, const std::string& name_out, float scalar, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int total = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    add_scalar_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, scalar, out_elem.ptr, total
    );
}

void CudaManager::add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& A = _get_element_no_lock(name_A);
    const Gpu_Element& B = _get_element_no_lock(name_B);
    const Gpu_Element& C = _get_element_no_lock(name_C);
    int total_elements = _get_total_elements(C);

    int blockSize = 256;
    int numBlocks = 4096; // Saturate GPU
    add_kernel_optimized<<<numBlocks, blockSize, 0, stream>>>(A.ptr, B.ptr, C.ptr, total_elements);
}

void CudaManager::add_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& src = _get_element_no_lock(name_src);
    const Gpu_Element& dest = _get_element_no_lock(name_dest);
    const int total_elements = _get_total_elements(src);

    if (total_elements > 0) {
        constexpr int blockSize = 256;
        const int numBlocks = (total_elements + blockSize - 1) / blockSize;
        add_into_kernel<<<numBlocks, blockSize, 0, stream>>>((const float*)src.ptr, (float*)dest.ptr, total_elements);
    }
}

void CudaManager::subtract(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& A = _get_element_no_lock(name_A);
    int n = _get_total_elements(A);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    elementwise_subtract_kernel<<<numBlocks, blockSize, 0, stream>>>(_get_element_no_lock(name_A).ptr, _get_element_no_lock(name_B).ptr, _get_element_no_lock(name_C).ptr, n);
}

void CudaManager::subtract_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& src = _get_element_no_lock(name_src);
    const Gpu_Element& dest = _get_element_no_lock(name_dest);
    int n = _get_total_elements(src);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    subtract_into_kernel<<<numBlocks, blockSize, 0, stream>>>(src.ptr, dest.ptr, n);
}

void CudaManager::multiply(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& A = _get_element_no_lock(name_A);
    int n = _get_total_elements(A);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    elementwise_multiply_kernel<<<numBlocks, blockSize, 0, stream>>>(_get_element_no_lock(name_A).ptr, _get_element_no_lock(name_B).ptr, _get_element_no_lock(name_C).ptr, n);
}
void CudaManager::multiply_backward(const std::string& grad_out_name, const std::string& other_input_name, const std::string& grad_in_name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out = _get_element_no_lock(grad_out_name);
    int n = _get_total_elements(grad_out);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    multiply_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out.ptr, _get_element_no_lock(other_input_name).ptr, _get_element_no_lock(grad_in_name).ptr, n);
}

void CudaManager::divide(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& A = _get_element_no_lock(name_A);
    int n = _get_total_elements(A);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    elementwise_divide_kernel<<<numBlocks, blockSize, 0, stream>>>(_get_element_no_lock(name_A).ptr, _get_element_no_lock(name_B).ptr, _get_element_no_lock(name_C).ptr, n);
}
void CudaManager::divide_backward(const std::string& A_name, const std::string& B_name, const std::string& grad_out_name, const std::string& grad_A_name, const std::string& grad_B_name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& A = _get_element_no_lock(A_name);
    int n = _get_total_elements(A);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    divide_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(A.ptr, _get_element_no_lock(B_name).ptr, _get_element_no_lock(grad_out_name).ptr, _get_element_no_lock(grad_A_name).ptr, _get_element_no_lock(grad_B_name).ptr, n);
}

void CudaManager::exp_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    exp_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::exp_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& out_data_elem = _get_element_no_lock(name_out_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    exp_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, out_data_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::log_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    log_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::log_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& in_data_elem = _get_element_no_lock(name_in_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    log_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, in_data_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::abs_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    abs_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::abs_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& in_data_elem = _get_element_no_lock(name_in_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    abs_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, in_data_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::sqrt_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sqrt_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::sqrt_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& out_data_elem = _get_element_no_lock(name_out_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sqrt_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, out_data_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::pow_elementwise(const std::string& name_in, const std::string& name_out, float exponent, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    pow_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, exponent, n);
}
void CudaManager::pow_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, float exponent, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& in_data_elem = _get_element_no_lock(name_in_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    pow_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, in_data_elem.ptr, grad_in_elem.ptr, exponent, n);
}

void CudaManager::clamp_elementwise(const std::string& name_in, const std::string& name_out, float min_val, float max_val, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    clamp_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, min_val, max_val, n);
}
void CudaManager::clamp_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, float min_val, float max_val, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& in_data_elem = _get_element_no_lock(name_in_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(grad_out_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    clamp_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, in_data_elem.ptr, grad_in_elem.ptr, min_val, max_val, n);
}


////////////////////////////////////////////////////////////////////////////////
///                          Matrix Operations                                //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::matmul(const std::string& name_A, const std::string& name_B, const std::string& name_C,bool transpose_A, bool transpose_B,float alpha, float beta,bool use_tensor_cores,cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    cublasSetStream(this->cublas_handle, stream);

    if (use_tensor_cores) {
        cublasSetMathMode(this->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    const Gpu_Element& A_elem = _get_element_no_lock(name_A);
    const Gpu_Element& B_elem = _get_element_no_lock(name_B);
    const Gpu_Element& C_elem = _get_element_no_lock(name_C);

    int M_A = A_elem.sizes[0];
    int K_A = A_elem.sizes[1];               // ← A must always be 2D, this is fine

    // ↓ REPLACE THE OLD   int M_B = B_elem.sizes[0]; int K_B = B_elem.sizes[1];  WITH THIS:
    int M_B, K_B;
    if (B_elem.sizes.size() == 1) {
        M_B = B_elem.sizes[0];
        K_B = 1;
    } else {
        M_B = B_elem.sizes[0];
        K_B = B_elem.sizes[1];
    }

    int m = transpose_A ? K_A : M_A;
    int k1 = transpose_A ? M_A : K_A;
    int k2 = transpose_B ? K_B : M_B;
    int n = transpose_B ? M_B : K_B;

    if (k1 != k2) {
        throw std::runtime_error("Matrix multiplication inner dimensions do not match.");
    }
    int k = k1;

    cublasOperation_t op_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(this->cublas_handle, op_B, op_A, n, m, k, &alpha,
                B_elem.ptr, K_B,
                A_elem.ptr, K_A,
                &beta, C_elem.ptr, n);

    if (use_tensor_cores) {
        cublasSetMathMode(this->cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

void CudaManager::transpose(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int width = in_elem.sizes[1];
    int height = in_elem.sizes[0];

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    transpose_shared_mem_kernel<<<grid, block, 0, stream>>>(in_elem.ptr, out_elem.ptr, width, height);
}

void CudaManager::broadcast_add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& A = _get_element_no_lock(name_A);
    const Gpu_Element& B = _get_element_no_lock(name_B);
    const Gpu_Element& C = _get_element_no_lock(name_C);

    int M = C.sizes[0];
    int N = C.sizes[1];

    int M_b = B.sizes.size() == 1 ? 1 : B.sizes[0];
    int N_b = B.sizes.size() == 1 ? B.sizes[0] : B.sizes[1];

    int b_type = -1;
    if (M_b == 1 && N_b == 1) b_type = 0;
    else if (M_b == 1 && N_b == N) b_type = 1;
    else if ((M_b == M && N_b == 1) || (M_b == 1 && N_b == M)) b_type = 2; // Allows Vector[M] to act as row bias

    if (b_type == -1) {
        throw std::runtime_error("broadcast_add: Tensor B shape mismatch.");
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    broadcast_add_kernel<<<grid, block, 0, stream>>>(A.ptr, B.ptr, C.ptr, M, N, b_type);
}



////////////////////////////////////////////////////////////////////////////////
///                         Activation Functions                              //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::relu(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int total_elements = _get_total_elements(in_elem);
    int blockSize = std::min(256, total_elements);
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, total_elements);
}
void CudaManager::relu_backward(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int total_elements = _get_total_elements(in_elem);
    int blockSize = std::min(256, total_elements);
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    relu_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, grad_out_elem.ptr, grad_in_elem.ptr, total_elements);
}

void CudaManager::sigmoid(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sigmoid_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::sigmoid_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& out_data_elem = _get_element_no_lock(name_out_data);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(out_data_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sigmoid_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(out_data_elem.ptr, grad_out_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::tanh(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    tanh_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::tanh_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& out_data_elem = _get_element_no_lock(name_out_data);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(out_data_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    tanh_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(out_data_elem.ptr, grad_out_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::gelu_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, n);
}
void CudaManager::gelu_backward(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, grad_out_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::softmax_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int M_rows = in_elem.sizes[0];
    int N_cols = in_elem.sizes[1];

    // Find the nearest power of 2 for the block size (Caps at 1024)
    // This safely handles non-power-of-2 sequence lengths!
    int power2_block = 1;
    while (power2_block < N_cols && power2_block < 1024) {
        power2_block *= 2;
    }

    dim3 grid(M_rows);
    dim3 block(power2_block);
    size_t shared_mem_size = block.x * sizeof(float);

    softmax_forward_kernel<<<grid, block, shared_mem_size, stream>>>(in_elem.ptr, out_elem.ptr, M_rows, N_cols);
}
void CudaManager::softmax_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& out_data_elem = _get_element_no_lock(name_out_data);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int M_rows = grad_out_elem.sizes[0];
    int N_cols = grad_out_elem.sizes[1];

    int power2_block = 1;
    while (power2_block < N_cols && power2_block < 1024) {
        power2_block *= 2;
    }

    dim3 grid(M_rows);
    dim3 block(power2_block);
    size_t shared_mem_size = block.x * sizeof(float);

    softmax_backward_kernel<<<grid, block, shared_mem_size, stream>>>(grad_out_elem.ptr, out_data_elem.ptr, grad_in_elem.ptr, M_rows, N_cols);
}


////////////////////////////////////////////////////////////////////////////////
///                            Loss Functions                                 //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::mse_loss_forward(const std::string& pred_name, const std::string& target_name, const std::string& out_name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& pred_elem = _get_element_no_lock(pred_name);
    const Gpu_Element& target_elem = _get_element_no_lock(target_name);
    const Gpu_Element& out_elem = _get_element_no_lock(out_name);
    int n = _get_total_elements(pred_elem);

    // 1. Squared Errors
    Gpu_Element squared_errors_elem;
    cudaMalloc(&squared_errors_elem.ptr, n * sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    elementwise_squared_error_kernel<<<numBlocks, blockSize, 0, stream>>>(pred_elem.ptr, target_elem.ptr, squared_errors_elem.ptr, n);

    // 2. Reduction (Sum) via CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, squared_errors_elem.ptr, out_elem.ptr, n, stream);

    Gpu_Element temp_cub_storage_elem;
    cudaMalloc(&temp_cub_storage_elem.ptr, temp_storage_bytes);

    cub::DeviceReduce::Sum(temp_cub_storage_elem.ptr, temp_storage_bytes, squared_errors_elem.ptr, out_elem.ptr, n, stream);

    // 3. Scale (Mean)
    scale_kernel<<<1, 1, 0, stream>>>(out_elem.ptr, 1.0f / static_cast<float>(n), 1);

    cudaFree(squared_errors_elem.ptr);
    cudaFree(temp_cub_storage_elem.ptr);
}
void CudaManager::mse_loss_backward(const std::string& pred_name, const std::string& target_name, const std::string& grad_in_name, const std::string& grad_out_name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& pred_elem     = _get_element_no_lock(pred_name);
    const Gpu_Element& target_elem   = _get_element_no_lock(target_name);
    const Gpu_Element& grad_in_elem  = _get_element_no_lock(grad_in_name);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(grad_out_name);

    int n = _get_total_elements(pred_elem);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    mse_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        pred_elem.ptr,
        target_elem.ptr,
        grad_out_elem.ptr,
        grad_in_elem.ptr,  // ← pass pointer directly, kernel reads grad_in[0]
        n
    );
}

void CudaManager::bce_loss_forward(const std::string& name_pred, const std::string& name_target, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& pred_elem = _get_element_no_lock(name_pred);
    const Gpu_Element& target_elem = _get_element_no_lock(name_target);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(pred_elem);

    // Temp buffer for element-wise errors
    Gpu_Element elementwise_errors_elem;
    cudaMalloc(&elementwise_errors_elem.ptr, n * sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    bce_elementwise_kernel<<<numBlocks, blockSize, 0, stream>>>(pred_elem.ptr, target_elem.ptr, elementwise_errors_elem.ptr, n);

    // CUB Reduction
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, elementwise_errors_elem.ptr, out_elem.ptr, n, stream);

    Gpu_Element temp_cub_storage_elem;
    cudaMalloc(&temp_cub_storage_elem.ptr, temp_storage_bytes);

    cub::DeviceReduce::Sum(temp_cub_storage_elem.ptr, temp_storage_bytes, elementwise_errors_elem.ptr, out_elem.ptr, n, stream);

    // Mean Scaling
    scale_kernel<<<1, 1, 0, stream>>>(out_elem.ptr, 1.0f / static_cast<float>(n), 1);

    cudaFree(elementwise_errors_elem.ptr);
    cudaFree(temp_cub_storage_elem.ptr);
}
void CudaManager::bce_loss_backward(const std::string& name_grad_out, const std::string& name_pred, const std::string& name_target, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& pred_elem = _get_element_no_lock(name_pred);
    const Gpu_Element& target_elem = _get_element_no_lock(name_target);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    int n = _get_total_elements(pred_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    bce_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, pred_elem.ptr, target_elem.ptr, grad_in_elem.ptr, n);
}

////////////////////////////////////////////////////////////////////////////////
///                             Optimizers                                    //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::sgd_update(const std::string& data_name, const std::string& grad_name, float learning_rate, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& data_elem = _get_element_no_lock(data_name);
    const Gpu_Element& grad_elem = _get_element_no_lock(grad_name);
    int n = _get_total_elements(data_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    sgd_update_kernel<<<numBlocks, blockSize, 0, stream>>>(data_elem.ptr, grad_elem.ptr, learning_rate, n);
}

void CudaManager::adam_update(const std::string& data_name,const std::string& grad_name,const std::string& m_name,const std::string& v_name,float learning_rate,float beta1,float beta2,float eps,int step,float weight_decay,cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    if (!exists(data_name) || !exists(grad_name) || !exists(m_name) || !exists(v_name)) {
        throw std::runtime_error("adam_update: one or more input tensors do not exist");
    }

    const Gpu_Element& data_elem = _get_element_no_lock(data_name);
    const Gpu_Element& grad_elem = _get_element_no_lock(grad_name);
    const Gpu_Element& m_elem    = _get_element_no_lock(m_name);
    const Gpu_Element& v_elem    = _get_element_no_lock(v_name);

    if (data_elem.sizes != grad_elem.sizes || data_elem.sizes != m_elem.sizes || data_elem.sizes != v_elem.sizes) {
        throw std::runtime_error("adam_update: size mismatch among data/grad/m/v tensors");
    }

    int n = _get_total_elements(data_elem);
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float b1t = std::pow(beta1, step);
    float b2t = std::pow(beta2, step);
    float one_minus_b1t = 1.0f - b1t;
    float one_minus_b2t = 1.0f - b2t;

    adam_update_kernel<<<numBlocks, blockSize, 0, stream>>>(
        data_elem.ptr, grad_elem.ptr, m_elem.ptr, v_elem.ptr,
        learning_rate, beta1, beta2, eps, one_minus_b1t, one_minus_b2t, weight_decay, n
    );
}

void CudaManager::clip_grad_value(const std::string& buffer_name, float clip_value, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& element = _get_element_no_lock(buffer_name);
    int n = _get_total_elements(element);
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    clip_grad_value_kernel<<<numBlocks, blockSize, 0, stream>>>(element.ptr, clip_value, n);
}


////////////////////////////////////////////////////////////////////////////////
///                    Advanced Layers & Fused Ops                            //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::matmul_bias_relu_forward(const std::string& name_X,const std::string& name_W,const std::string& name_B,const std::string& name_Relu_Out,const std::string& name_PreRelu_Out,cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    const Gpu_Element& X_elem = _get_element_no_lock(name_X);
    const Gpu_Element& W_elem = _get_element_no_lock(name_W);
    const Gpu_Element& B_elem = _get_element_no_lock(name_B);
    const Gpu_Element& Relu_Out_elem = _get_element_no_lock(name_Relu_Out);
    const Gpu_Element& PreRelu_Out_elem = _get_element_no_lock(name_PreRelu_Out);

    if (X_elem.sizes.size() != 2 || W_elem.sizes.size() != 2 || X_elem.sizes[1] != W_elem.sizes[0]) {
        throw std::runtime_error("matmul_bias_relu: Invalid matrix dimensions for X@W.");
    }
    int M = X_elem.sizes[0];
    int K = X_elem.sizes[1];
    int N = W_elem.sizes[1];

    if (Relu_Out_elem.sizes != std::vector<int>{M, N} || PreRelu_Out_elem.sizes != std::vector<int>{M, N}) {
          throw std::runtime_error("matmul_bias_relu: Output dimension mismatch.");
    }

    // Matmul (PreRelu_Out = X @ W)
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSetStream(this->cublas_handle, stream);

    // --- ENABLE TENSOR CORES ---
    cublasSetMathMode(this->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasSgemm(this->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, W_elem.ptr, N, X_elem.ptr, K,
                &beta, PreRelu_Out_elem.ptr, N);

    // --- REVERT ---
    cublasSetMathMode(this->cublas_handle, CUBLAS_DEFAULT_MATH);

    // Fused Bias + ReLU
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    add_bias_relu_store_pre_relu_kernel<<<gridSize, blockSize, 0, stream>>>(
        (const float*)PreRelu_Out_elem.ptr,
        B_elem.ptr,
        Relu_Out_elem.ptr,
        PreRelu_Out_elem.ptr,
        M, N
    );
}

void CudaManager::layer_norm_forward(const std::string& name_in, const std::string& name_gamma, const std::string& name_beta,const std::string& name_out, const std::string& name_mean, const std::string& name_rstd,float epsilon, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& gamma_elem = _get_element_no_lock(name_gamma);
    const Gpu_Element& beta_elem = _get_element_no_lock(name_beta);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    const Gpu_Element& mean_elem = _get_element_no_lock(name_mean);
    const Gpu_Element& rstd_elem = _get_element_no_lock(name_rstd);

    int M_rows = in_elem.sizes[0];
    int N_cols = in_elem.sizes[1];

    int threads = 256; // Safe power of 2
    dim3 grid(M_rows);
    dim3 block(threads);
    size_t shared_mem_size = threads * sizeof(float);

    layer_norm_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        in_elem.ptr, gamma_elem.ptr, beta_elem.ptr,
        out_elem.ptr, mean_elem.ptr, rstd_elem.ptr,
        M_rows, N_cols, epsilon
    );
}
void CudaManager::layer_norm_backward(const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma,const std::string& name_mean, const std::string& name_rstd,const std::string& name_grad_in, const std::string& name_grad_gamma, const std::string& name_grad_beta,cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& gamma_elem = _get_element_no_lock(name_gamma);
    const Gpu_Element& mean_elem = _get_element_no_lock(name_mean);
    const Gpu_Element& rstd_elem = _get_element_no_lock(name_rstd);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    const Gpu_Element& grad_gamma_elem = _get_element_no_lock(name_grad_gamma);
    const Gpu_Element& grad_beta_elem = _get_element_no_lock(name_grad_beta);

    int M_rows = in_elem.sizes[0];
    int N_cols = in_elem.sizes[1];

    int threads = 256; // Safe power of 2
    dim3 grid(M_rows);
    dim3 block(threads);
    size_t shared_mem_size = threads * 2 * sizeof(float);

    layer_norm_backward_kernel<<<grid, block, shared_mem_size, stream>>>(
        grad_out_elem.ptr, in_elem.ptr, gamma_elem.ptr,
        mean_elem.ptr, rstd_elem.ptr,
        grad_in_elem.ptr, grad_gamma_elem.ptr, grad_beta_elem.ptr,
        M_rows, N_cols
    );
}

void CudaManager::conv2d_forward(const std::string& name_in, const std::string& name_kernel, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& kernel_elem = _get_element_no_lock(name_kernel);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int inHeight = in_elem.sizes[0], inWidth = in_elem.sizes[1];
    int kernelHeight = kernel_elem.sizes[0], kernelWidth = kernel_elem.sizes[1];
    int outHeight = out_elem.sizes[0], outWidth = out_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    conv2d_forward_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, kernel_elem.ptr, out_elem.ptr,
        inHeight, inWidth, kernelHeight, kernelWidth, outHeight, outWidth
    );
}
void CudaManager::conv2d_backward_input(const std::string& name_grad_out, const std::string& name_kernel, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& kernel_elem = _get_element_no_lock(name_kernel);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int inHeight = grad_in_elem.sizes[0], inWidth = grad_in_elem.sizes[1];
    int kernelHeight = kernel_elem.sizes[0], kernelWidth = kernel_elem.sizes[1];
    int outHeight = grad_out_elem.sizes[0], outWidth = grad_out_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    conv2d_backward_input_kernel<<<grid, block, 0, stream>>>(
        grad_out_elem.ptr, kernel_elem.ptr, grad_in_elem.ptr,
        inHeight, inWidth, kernelHeight, kernelWidth, outHeight, outWidth
    );
}
void CudaManager::conv2d_backward_kernel(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_kernel, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_kernel_elem = _get_element_no_lock(name_grad_kernel);

    int inHeight = in_elem.sizes[0], inWidth = in_elem.sizes[1];
    int kernelHeight = grad_kernel_elem.sizes[0], kernelWidth = grad_kernel_elem.sizes[1];
    int outHeight = grad_out_elem.sizes[0], outWidth = grad_out_elem.sizes[1];

    dim3 block(kernelWidth, kernelHeight);
    dim3 grid(1, 1);

    if (kernelWidth * kernelHeight > 1024) {
        block.x = 16; block.y = 16;
        grid.x = (kernelWidth + block.x - 1) / block.x;
        grid.y = (kernelHeight + block.y - 1) / block.y;
    }

    conv2d_backward_kernel_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, grad_out_elem.ptr, grad_kernel_elem.ptr,
        inHeight, inWidth, kernelHeight, kernelWidth, outHeight, outWidth
    );
}

void CudaManager::conv2d_multi_forward(const std::string& name_in, const std::string& name_weight, const std::string& name_bias, const std::string& name_out,int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int in_h = (in_elem.sizes.size() == 3) ? in_elem.sizes[1] : in_elem.sizes[0];
    int in_w = (in_elem.sizes.size() == 3) ? in_elem.sizes[2] : in_elem.sizes[1];
    int out_h = out_elem.sizes[1];
    int out_w = out_elem.sizes[2];

    dim3 block(8, 8, 8);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y, (C_out + block.z - 1) / block.z);

    conv2d_multi_forward_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, _get_element_no_lock(name_weight).ptr, _get_element_no_lock(name_bias).ptr, out_elem.ptr,
        C_in, in_h, in_w, C_out, k_h, k_w, out_h, out_w, pad_t, pad_l, stride_h, stride_w
    );
}
void CudaManager::conv2d_multi_backward_input(const std::string& name_grad_out, const std::string& name_weight, const std::string& name_grad_in,int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);

    int in_h = (grad_in_elem.sizes.size() == 3) ? grad_in_elem.sizes[1] : grad_in_elem.sizes[0];
    int in_w = (grad_in_elem.sizes.size() == 3) ? grad_in_elem.sizes[2] : grad_in_elem.sizes[1];
    int out_h = grad_out_elem.sizes[1];
    int out_w = grad_out_elem.sizes[2];

    dim3 block(8, 8, 8);
    dim3 grid((in_w + block.x - 1) / block.x, (in_h + block.y - 1) / block.y, (C_in + block.z - 1) / block.z);

    conv2d_multi_backward_input_kernel<<<grid, block, 0, stream>>>(
        grad_out_elem.ptr, _get_element_no_lock(name_weight).ptr, grad_in_elem.ptr,
        C_in, in_h, in_w, C_out, k_h, k_w, out_h, out_w, pad_t, pad_l, stride_h, stride_w
    );
}
void CudaManager::conv2d_multi_backward_weight(const std::string& name_input, const std::string& name_grad_out, const std::string& name_grad_weight, const std::string& name_grad_bias,int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_input);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);

    int in_h = (in_elem.sizes.size() == 3) ? in_elem.sizes[1] : in_elem.sizes[0];
    int in_w = (in_elem.sizes.size() == 3) ? in_elem.sizes[2] : in_elem.sizes[1];
    int out_h = grad_out_elem.sizes[1];
    int out_w = grad_out_elem.sizes[2];

    int total_weights = C_out * C_in * k_h * k_w;
    int blockSize = 256;
    int numBlocksW = (total_weights + blockSize - 1) / blockSize;

    conv2d_multi_backward_weight_kernel<<<numBlocksW, blockSize, 0, stream>>>(
        in_elem.ptr, grad_out_elem.ptr, _get_element_no_lock(name_grad_weight).ptr,
        C_in, in_h, in_w, C_out, k_h, k_w, out_h, out_w, pad_t, pad_l, stride_h, stride_w
    );

    int numBlocksB = (C_out + blockSize - 1) / blockSize;
    conv2d_multi_backward_bias_kernel<<<numBlocksB, blockSize, 0, stream>>>(
        grad_out_elem.ptr, _get_element_no_lock(name_grad_bias).ptr, C_out, out_h, out_w
    );
}

void CudaManager::im2col(const std::string& name_in, const std::string& name_out_col, int kernelHeight, int kernelWidth, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_col_elem = _get_element_no_lock(name_out_col);

    int inHeight = in_elem.sizes[0];
    int inWidth = in_elem.sizes[1];
    int outHeight = inHeight - kernelHeight + 1;
    int outWidth = inWidth - kernelWidth + 1;

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    im2col_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, out_col_elem.ptr,
        inHeight, inWidth, kernelHeight, kernelWidth, outHeight, outWidth
    );
}
void CudaManager::col2im(const std::string& name_col_grad, const std::string& name_in_grad, int kernelHeight, int kernelWidth, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& col_grad_elem = _get_element_no_lock(name_col_grad);
    const Gpu_Element& in_grad_elem = _get_element_no_lock(name_in_grad);

    int inHeight = in_grad_elem.sizes[0];
    int inWidth = in_grad_elem.sizes[1];
    int outHeight = inHeight - kernelHeight + 1;
    int outWidth = inWidth - kernelWidth + 1;

    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    col2im_kernel<<<grid, block, 0, stream>>>(
        col_grad_elem.ptr, in_grad_elem.ptr,
        inHeight, inWidth, kernelHeight, kernelWidth, outHeight, outWidth
    );
}

void CudaManager::global_avg_pool_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int seq_len = in_elem.sizes[0];
    int d_model = in_elem.sizes[1];

    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;

    global_avg_pool_forward_kernel<<<blocks, threads, 0, stream>>>(in_elem.ptr, out_elem.ptr, seq_len, d_model);
}
void CudaManager::global_avg_pool_backward(const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int seq_len = grad_in_elem.sizes[0];
    int d_model = grad_in_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((d_model + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    global_avg_pool_backward_kernel<<<grid, block, 0, stream>>>(grad_out_elem.ptr, grad_in_elem.ptr, seq_len, d_model);
}

void CudaManager::max_pool_1d_forward(const std::string& name_in, const std::string& name_out, const std::string& name_indices, int pool_size, int stride, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);

    int in_len = in_elem.sizes[0];
    int out_len = out_elem.sizes[0];

    int blockSize = 256;
    int numBlocks = (out_len + blockSize - 1) / blockSize;

    max_pool_1d_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, indices_elem.ptr,
        in_len, out_len, pool_size, stride
    );
}
void CudaManager::max_pool_1d_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int out_len = grad_out_elem.sizes[0];

    int blockSize = 256;
    int numBlocks = (out_len + blockSize - 1) / blockSize;

    max_pool_1d_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_out_elem.ptr, indices_elem.ptr, grad_in_elem.ptr, out_len
    );
}

void CudaManager::max_pool_2d_forward(const std::string& name_in, const std::string& name_out, const std::string& name_indices, int pool_size, int stride, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);

    int in_h = in_elem.sizes[0];
    int in_w = in_elem.sizes[1];
    int out_h = out_elem.sizes[0];
    int out_w = out_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    max_pool_2d_forward_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, indices_elem.ptr,
        in_h, in_w, out_h, out_w, pool_size, stride
    );
}
void CudaManager::max_pool_2d_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int out_total = _get_total_elements(grad_out_elem);

    int blockSize = 256;
    int numBlocks = (out_total + blockSize - 1) / blockSize;

    // We can reuse the 1D backward kernel structure since indices are flattened!
    max_pool_2d_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_out_elem.ptr, indices_elem.ptr, grad_in_elem.ptr, out_total
    );
}

void CudaManager::avg_pool_2d_forward(const std::string& name_in, const std::string& name_out, int pool_size, int stride, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int in_h = in_elem.sizes[0];
    int in_w = in_elem.sizes[1];
    int out_h = out_elem.sizes[0];
    int out_w = out_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    avg_pool_2d_forward_kernel<<<grid, block, 0, stream>>>(
        in_elem.ptr, out_elem.ptr,
        in_h, in_w, out_h, out_w, pool_size, stride
    );
}
void CudaManager::avg_pool_2d_backward(const std::string& name_grad_out, const std::string& name_grad_in, int pool_size, int stride, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int in_h = grad_in_elem.sizes[0];
    int in_w = grad_in_elem.sizes[1];
    int out_h = grad_out_elem.sizes[0];
    int out_w = grad_out_elem.sizes[1];

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    avg_pool_2d_backward_kernel<<<grid, block, 0, stream>>>(
        grad_out_elem.ptr, grad_in_elem.ptr,
        in_h, in_w, out_h, out_w, pool_size, stride
    );
}

void CudaManager::batch_norm_1d_forward(const std::string& name_in, const std::string& name_gamma, const std::string& name_beta, const std::string& name_running_mean, const std::string& name_running_var, const std::string& name_out, const std::string& name_saved_mean, const std::string& name_saved_inv_var, float momentum, float epsilon, bool is_training, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);

    int num_features = in_elem.sizes[0];
    int blockSize = 256;
    int numBlocks = (num_features + blockSize - 1) / blockSize;

    batch_norm_1d_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, _get_element_no_lock(name_gamma).ptr, _get_element_no_lock(name_beta).ptr,
        _get_element_no_lock(name_running_mean).ptr, _get_element_no_lock(name_running_var).ptr,
        _get_element_no_lock(name_out).ptr, _get_element_no_lock(name_saved_mean).ptr, _get_element_no_lock(name_saved_inv_var).ptr,
        num_features, momentum, epsilon, is_training
    );
}
void CudaManager::batch_norm_1d_backward(const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma, const std::string& name_saved_mean, const std::string& name_saved_inv_var, const std::string& name_grad_in, const std::string& name_grad_gamma, const std::string& name_grad_beta, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);

    int num_features = in_elem.sizes[0];
    int blockSize = 256;
    int numBlocks = (num_features + blockSize - 1) / blockSize;

    batch_norm_1d_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        _get_element_no_lock(name_grad_out).ptr, in_elem.ptr, _get_element_no_lock(name_gamma).ptr,
        _get_element_no_lock(name_saved_mean).ptr, _get_element_no_lock(name_saved_inv_var).ptr,
        _get_element_no_lock(name_grad_in).ptr, _get_element_no_lock(name_grad_gamma).ptr, _get_element_no_lock(name_grad_beta).ptr,
        num_features
    );
}

void CudaManager::batch_norm_2d_forward(const std::string& name_in, const std::string& name_gamma, const std::string& name_beta, const std::string& name_running_mean, const std::string& name_running_var, const std::string& name_out, const std::string& name_saved_mean, const std::string& name_saved_inv_var, float momentum, float epsilon, bool is_training, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);

    int num_channels = in_elem.sizes[0];
    int height = in_elem.sizes[1];
    int width = in_elem.sizes[2];
    int elements_per_channel = height * width;

    dim3 grid(num_channels);
    dim3 block(256);
    if (elements_per_channel < 256) block.x = elements_per_channel;

    size_t shared_mem_size = block.x * sizeof(float);

    batch_norm_2d_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        in_elem.ptr, _get_element_no_lock(name_gamma).ptr, _get_element_no_lock(name_beta).ptr,
        _get_element_no_lock(name_running_mean).ptr, _get_element_no_lock(name_running_var).ptr,
        _get_element_no_lock(name_out).ptr, _get_element_no_lock(name_saved_mean).ptr, _get_element_no_lock(name_saved_inv_var).ptr,
        num_channels, elements_per_channel, momentum, epsilon, is_training
    );
}
void CudaManager::batch_norm_2d_backward(const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma, const std::string& name_saved_mean, const std::string& name_saved_inv_var, const std::string& name_grad_in, const std::string& name_grad_gamma, const std::string& name_grad_beta, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);

    int num_channels = in_elem.sizes[0];
    int height = in_elem.sizes[1];
    int width = in_elem.sizes[2];
    int elements_per_channel = height * width;

    dim3 grid(num_channels);
    dim3 block(256);
    if (elements_per_channel < 256) block.x = elements_per_channel;

    size_t shared_mem_size = block.x * 2 * sizeof(float);

    batch_norm_2d_backward_kernel<<<grid, block, shared_mem_size, stream>>>(
        _get_element_no_lock(name_grad_out).ptr, in_elem.ptr, _get_element_no_lock(name_gamma).ptr,
        _get_element_no_lock(name_saved_mean).ptr, _get_element_no_lock(name_saved_inv_var).ptr,
        _get_element_no_lock(name_grad_in).ptr, _get_element_no_lock(name_grad_gamma).ptr, _get_element_no_lock(name_grad_beta).ptr,
        num_channels, elements_per_channel
    );
}

void CudaManager::scale_matrix(const std::string& name_in, const std::string& name_out, float scale, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int size = 1;
    for (int dim : in_elem.sizes) {
        size = size * dim;
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    scale_matrix_forward_kernel<<<blocks, threads, 0, stream>>>(in_elem.ptr, out_elem.ptr, scale, size);
}
void CudaManager::scale_matrix_backward(const std::string& name_grad_out, const std::string& name_grad_in, float scale, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int size = 1;
    for (int dim : grad_out_elem.sizes) {
        size = size * dim;
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    scale_matrix_backward_kernel<<<blocks, threads, 0, stream>>>(grad_out_elem.ptr, grad_in_elem.ptr, scale, size);
}

void CudaManager::dropout_forward(const std::string& name_in, const std::string& name_out, const std::string& name_mask, float drop_rate, int seed, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    const Gpu_Element& mask_elem = _get_element_no_lock(name_mask);

    int n = _get_total_elements(in_elem);
    if (n == 0) return;

    float scale = 1.0f / (1.0f - drop_rate);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    dropout_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, mask_elem.ptr, drop_rate, scale, static_cast<unsigned int>(seed), n
    );
}
void CudaManager::dropout_backward(const std::string& name_grad_out, const std::string& name_mask, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& mask_elem = _get_element_no_lock(name_mask);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    int n = _get_total_elements(grad_out_elem);
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    dropout_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_out_elem.ptr, mask_elem.ptr, grad_in_elem.ptr, n
    );
}

void CudaManager::markov_count(const std::string& name_sequence, const std::string& name_count_table, int order, int num_states, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& seq_elem = _get_element_no_lock(name_sequence);
    const Gpu_Element& count_elem = _get_element_no_lock(name_count_table);

    int seq_len = _get_total_elements(seq_elem);
    if (seq_len <= order) return;

    int blockSize = 256;
    int numBlocks = ((seq_len - order) + blockSize - 1) / blockSize;
    markov_count_kernel<<<numBlocks, blockSize, 0, stream>>>(seq_elem.ptr, seq_len, count_elem.ptr, order, num_states);
}
void CudaManager::markov_normalize(const std::string& name_count_table, const std::string& name_prob_table, int num_histories, int num_states, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& count_elem = _get_element_no_lock(name_count_table);
    const Gpu_Element& prob_elem = _get_element_no_lock(name_prob_table);

    int blockSize = 256;
    int numBlocks = (num_histories + blockSize - 1) / blockSize;
    markov_normalize_kernel<<<numBlocks, blockSize, 0, stream>>>(count_elem.ptr, prob_elem.ptr, num_histories, num_states);
}
void CudaManager::markov_predict(const std::string& name_history, const std::string& name_prob_table, const std::string& name_out_probs, int order, int num_states, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& hist_elem = _get_element_no_lock(name_history);
    const Gpu_Element& prob_elem = _get_element_no_lock(name_prob_table);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out_probs);

    int batch_size = hist_elem.sizes[0];
    if (batch_size == 0) return;

    // Launch one thread per OUTPUT float, not per batch item
    int total_elements = batch_size * num_states;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    markov_predict_kernel<<<numBlocks, blockSize, 0, stream>>>(hist_elem.ptr, prob_elem.ptr, out_elem.ptr, batch_size, order, num_states);
}

////////////////////////////////////////////////////////////////////////////////
///                        Tensor Manipulation                                //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::slice_row(const std::string& name_in, const std::string& name_out, int row_index, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem  = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int in_cols = in_elem.sizes[1];  // matrix, always 2D, fine

    // out is a 1D vector — use sizes[0] instead of sizes[1]
    int n = (out_elem.sizes.size() == 1) ? out_elem.sizes[0] : out_elem.sizes[1];

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    slice_row_kernel<<<numBlocks, blockSize, 0, stream>>>(in_elem.ptr, out_elem.ptr, in_cols, row_index);
}
void CudaManager::slice_row_backward(const std::string& name_grad_out, const std::string& name_grad_in, int row_index, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem  = _get_element_no_lock(name_grad_in);

    int in_cols = grad_in_elem.sizes[1];  // matrix grad, always 2D, fine

    // grad_out is a 1D vector
    int n = (grad_out_elem.sizes.size() == 1) ? grad_out_elem.sizes[0] : grad_out_elem.sizes[1];

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    slice_row_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, grad_in_elem.ptr, in_cols, row_index);
}

void CudaManager::slice_column(const std::string& name_in, const std::string& name_out, int start_col, int end_col, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    if (in_elem.sizes.size() != 2 || out_elem.sizes.size() != 2) {
        throw std::runtime_error("slice_column: Tensors must be 2D.");
    }

    int in_rows = in_elem.sizes[0];
    int in_cols_total = in_elem.sizes[1];
    int out_rows = out_elem.sizes[0];
    int out_cols = out_elem.sizes[1];

    if (in_rows != out_rows) throw std::runtime_error("slice_column: row dimension mismatch.");
    if (start_col < 0 || end_col > in_cols_total || start_col >= end_col) throw std::runtime_error("slice_column: invalid column indices.");
    if (out_cols != (end_col - start_col)) throw std::runtime_error("slice_column: output column size does not match slice range.");

    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x, (in_rows + block.y - 1) / block.y);

    slice_column_kernel<<<grid, block, 0, stream>>>(in_elem.ptr, out_elem.ptr, in_rows, in_cols_total, out_cols, start_col);
}
void CudaManager::slice_column_backward(const std::string& name_grad_out, const std::string& name_grad_in, int start_col, int end_col, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in);

    if (grad_in_elem.sizes.size() != 2 || grad_out_elem.sizes.size() != 2) {
        throw std::runtime_error("slice_column_backward: Tensors must be 2D.");
    }

    int in_rows = grad_in_elem.sizes[0];
    int in_cols_total = grad_in_elem.sizes[1];
    int out_rows = grad_out_elem.sizes[0];
    int out_cols = grad_out_elem.sizes[1];

    if (in_rows != out_rows) throw std::runtime_error("slice_column_backward: row dimension mismatch.");
    if (start_col < 0 || end_col > in_cols_total || start_col >= end_col) throw std::runtime_error("slice_column_backward: invalid column indices.");
    if (out_cols != (end_col - start_col)) throw std::runtime_error("slice_column_backward: output column size does not match slice range.");

    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x, (in_rows + block.y - 1) / block.y);

    slice_column_backward_kernel<<<grid, block, 0, stream>>>(grad_out_elem.ptr, grad_in_elem.ptr, in_rows, in_cols_total, out_cols, start_col);
}

void CudaManager::stack_rows(const std::vector<std::string>& names_in, const std::string& name_out, int axis, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int offset = 0;
    for (const std::string& name : names_in) {
        const Gpu_Element& in_elem = _get_element_no_lock(name);
        int total_elements = _get_total_elements(in_elem);

        // Pure contiguous memory copy - completely ignores shape
        CHECK_CUDA(cudaMemcpyAsync(
            out_elem.ptr + offset,
            in_elem.ptr,
            total_elements * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        offset += total_elements;
    }
}

void CudaManager::stack_rows_backward(const std::string& name_grad_out, const std::vector<std::string>& names_grad_in, int axis, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);

    int offset = 0;
    for (const std::string& name : names_grad_in) {
        const Gpu_Element& grad_in_elem = _get_element_no_lock(name);
        int total_elements = _get_total_elements(grad_in_elem);

        // Scatter the gradients back to the original tensors
        CHECK_CUDA(cudaMemcpyAsync(
            grad_in_elem.ptr,
            grad_out_elem.ptr + offset,
            total_elements * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        offset += total_elements;
    }
}
void CudaManager::concatenate(const std::string& name_A, const std::string& name_B, const std::string& name_out, int axis, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);

    if (axis == 0) {
        const Gpu_Element& A = _get_element_no_lock(name_A);
        const Gpu_Element& B = _get_element_no_lock(name_B);
        const Gpu_Element& C = _get_element_no_lock(name_out);
        int n_A = _get_total_elements(A);
        int n_B = _get_total_elements(B);
        // Simple back-to-back copy
        CHECK_CUDA(cudaMemcpyAsync(C.ptr,        A.ptr, n_A * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(C.ptr + n_A,  B.ptr, n_B * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (axis != 1) throw std::runtime_error("concatenate: only axis=1 (column) is supported.");

    const Gpu_Element& A = _get_element_no_lock(name_A);
    const Gpu_Element& B = _get_element_no_lock(name_B);
    const Gpu_Element& C = _get_element_no_lock(name_out);

    if (A.sizes.size() != 2 || B.sizes.size() != 2 || C.sizes.size() != 2) throw std::runtime_error("concatenate: All tensors must be 2D.");
    if (A.sizes[0] != B.sizes[0] || A.sizes[0] != C.sizes[0]) throw std::runtime_error("concatenate: All tensors must have the same number of rows.");

    int Rows = A.sizes[0];
    int Cols_A = A.sizes[1];
    int Cols_B = B.sizes[1];
    int Cols_C = C.sizes[1];

    if (Cols_C != Cols_A + Cols_B) throw std::runtime_error("concatenate: Output columns (C) must equal A_cols + B_cols.");

    dim3 block(16, 16);
    dim3 grid((Cols_C + block.x - 1) / block.x, (Rows + block.y - 1) / block.y);

    concatenate_kernel<<<grid, block, 0, stream>>>(A.ptr, B.ptr, C.ptr, Rows, Cols_A, Cols_B, Cols_C);
}
void CudaManager::concatenate_backward(const std::string& name_grad_out, const std::string& name_grad_in_A, const std::string& name_grad_in_B, int axis, int split_index, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    if (axis == 0) {
        const Gpu_Element& grad_out = _get_element_no_lock(name_grad_out);
        const Gpu_Element& grad_A   = _get_element_no_lock(name_grad_in_A);
        const Gpu_Element& grad_B   = _get_element_no_lock(name_grad_in_B);

        int n_A = split_index;
        int n_B = _get_total_elements(grad_out) - split_index;

        int blockSize = 256;

        // grad_A += grad_out[0 . n_A-1]
        int numBlocksA = (n_A + blockSize - 1) / blockSize;
        add_into_kernel<<<numBlocksA, blockSize, 0, stream>>>(grad_out.ptr, grad_A.ptr, n_A);

        // grad_B += grad_out[n_A .. n_A+n_B-1]
        int numBlocksB = (n_B + blockSize - 1) / blockSize;
        add_into_kernel<<<numBlocksB, blockSize, 0, stream>>>(grad_out.ptr + n_A, grad_B.ptr, n_B);

        return;
    }

    if (axis != 1) throw std::runtime_error("concatenate_backward: only axis=1 (column) is supported.");

    const Gpu_Element& grad_out = _get_element_no_lock(name_grad_out);
    const Gpu_Element& grad_in_A = _get_element_no_lock(name_grad_in_A);
    const Gpu_Element& grad_in_B = _get_element_no_lock(name_grad_in_B);

    if (grad_out.sizes.size() != 2 || grad_in_A.sizes.size() != 2 || grad_in_B.sizes.size() != 2) throw std::runtime_error("concatenate_backward: All tensors must be 2D.");
    if (grad_out.sizes[0] != grad_in_A.sizes[0] || grad_out.sizes[0] != grad_in_B.sizes[0]) throw std::runtime_error("concatenate_backward: All tensors must have the same number of rows.");

    int Rows = grad_out.sizes[0];
    int Cols_A = grad_in_A.sizes[1];
    int Cols_B = grad_in_B.sizes[1];
    int Cols_C = grad_out.sizes[1];

    if (Cols_A != split_index) throw std::runtime_error("concatenate_backward: split_index attribute does not match grad_in_A's column size.");
    if (Cols_C != Cols_A + Cols_B) throw std::runtime_error("concatenate_backward: grad_out columns must equal grad_in_A_cols + grad_in_B_cols.");

    dim3 block(16, 16);
    dim3 grid((Cols_C + block.x - 1) / block.x, (Rows + block.y - 1) / block.y);

    concatenate_backward_kernel<<<grid, block, 0, stream>>>(grad_out.ptr, grad_in_A.ptr, grad_in_B.ptr, Rows, Cols_A, Cols_B, Cols_C);
}


////////////////////////////////////////////////////////////////////////////////
///                            Reduction Ops                                  //
////////////////////////////////////////////////////////////////////////////////

void CudaManager::sum_reduce(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);
    int n = _get_total_elements(in_elem);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, in_elem.ptr, out_elem.ptr, n, stream);

    Gpu_Element temp_cub_storage_elem;
    cudaMalloc(&temp_cub_storage_elem.ptr, temp_storage_bytes);

    cub::DeviceReduce::Sum(temp_cub_storage_elem.ptr, temp_storage_bytes, in_elem.ptr, out_elem.ptr, n, stream);

    cudaFree(temp_cub_storage_elem.ptr);
}
void CudaManager::sum_reduce_backward(const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out); // Scalar [1]
    const Gpu_Element& grad_in_elem = _get_element_no_lock(name_grad_in); // Matrix [M, N]
    int n = _get_total_elements(grad_in_elem);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sum_reduce_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(grad_out_elem.ptr, grad_in_elem.ptr, n);
}

void CudaManager::sum_reduce_columns(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem  = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    if (in_elem.sizes.size() != 2) {
        throw std::runtime_error("sum_reduce_columns: Input must be 2D.");
    }

    int rows = in_elem.sizes[0];
    int cols = in_elem.sizes[1];

    // Accept output as either [cols] (1D) or [1, cols] (2D row vector)
    int out_cols;
    if (out_elem.sizes.size() == 1) {
        out_cols = out_elem.sizes[0];
    } else if (out_elem.sizes.size() == 2) {
        out_cols = out_elem.sizes[1];
    } else {
        throw std::runtime_error("sum_reduce_columns: Output must be 1D [N] or 2D [1, N].");
    }

    if (out_cols != cols) {
        throw std::runtime_error("sum_reduce_columns: Output size must match input column count.");
    }

    int total_elements = rows * cols;
    CHECK_CUDA(cudaMemsetAsync(out_elem.ptr, 0, cols * sizeof(float), stream));

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    if (numBlocks > 4096) numBlocks = 4096;

    sum_reduce_columns_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, rows, cols, total_elements
    );
}
void CudaManager::sum_reduce_rows(const std::string& name_in, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& in_elem  = _get_element_no_lock(name_in);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int rows = in_elem.sizes[0];
    int cols = in_elem.sizes[1];
    int total_elements = rows * cols;

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    if (numBlocks > 4096) numBlocks = 4096;

    sum_reduce_rows_kernel<<<numBlocks, blockSize, 0, stream>>>(
        in_elem.ptr, out_elem.ptr, rows, cols, total_elements
    );
}

void CudaManager::embedding_forward(const std::string& name_indices, const std::string& name_weight, const std::string& name_out, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);
    const Gpu_Element& weight_elem = _get_element_no_lock(name_weight);
    const Gpu_Element& out_elem = _get_element_no_lock(name_out);

    int num_indices = _get_total_elements(indices_elem);
    int vocab_size = weight_elem.sizes[0];
    int embedding_dim = weight_elem.sizes[1];

    int total_threads = num_indices * embedding_dim;
    int blockSize = 256;
    int numBlocks = (total_threads + blockSize - 1) / blockSize;

    embedding_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        indices_elem.ptr, weight_elem.ptr, out_elem.ptr,
        num_indices, embedding_dim, vocab_size
    );
}
void CudaManager::embedding_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_weight, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(this->cuda_mutex);
    const Gpu_Element& grad_out_elem = _get_element_no_lock(name_grad_out);
    const Gpu_Element& indices_elem = _get_element_no_lock(name_indices);
    const Gpu_Element& grad_weight_elem = _get_element_no_lock(name_grad_weight);

    int num_indices = _get_total_elements(indices_elem);
    int vocab_size = grad_weight_elem.sizes[0];
    int embedding_dim = grad_weight_elem.sizes[1];

    int total_threads = num_indices * embedding_dim;
    int blockSize = 256;
    int numBlocks = (total_threads + blockSize - 1) / blockSize;

    embedding_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        grad_out_elem.ptr, indices_elem.ptr, grad_weight_elem.ptr,
        num_indices, embedding_dim, vocab_size
    );
}



