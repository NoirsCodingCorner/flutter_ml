#include <iostream>
#include <cublas_v2.h>

// This wrapper function will be called from Dart.
extern "C" __declspec(dllexport)
void multiplyMatrices(float* h_a, float* h_b, float* h_c, int m, int k, int n) {
    // A is an m x k matrix
    // B is a k x n matrix
    // C is the resulting m x n matrix

    float *d_a, *d_b, *d_c;
    int a_size = m * k * sizeof(float);
    int b_size = k * n * sizeof(float);
    int c_size = m * n * sizeof(float);

    // 1. Allocate memory on the GPU device
    cudaMalloc((void**)&d_a, a_size);
    cudaMalloc((void**)&d_b, b_size);
    cudaMalloc((void**)&d_c, c_size);

    // 2. Copy matrices from host (CPU) to device (GPU)
    cudaMemcpy(d_a, h_a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

    // 3. Use cuBLAS to perform the matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set scaling factors for C = alpha*(A*B) + beta*C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // The main cuBLAS call for Single-precision General Matrix Multiplication (SGEMM)
    // IMPORTANT: cuBLAS uses column-major order, while C++/Dart use row-major.
    // A standard trick is to compute C = B^T * A^T which is equivalent to C = A * B
    // in row-major. We do this by swapping A and B in the call and their dimensions.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_b, n,
                d_a, k,
                &beta,
                d_c, n);

    // Clean up the cuBLAS handle
    cublasDestroy(handle);

    // 4. Copy the result matrix C from device (GPU) back to host (CPU)
    cudaMemcpy(h_c, d_c, c_size, cudaMemcpyDeviceToHost);

    // 5. Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}