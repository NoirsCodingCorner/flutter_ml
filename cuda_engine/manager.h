#ifndef MANAGER_H
#define MANAGER_H

#include <cublas_v2.h>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <mutex>

#define CHECK_CUDA(val) checkCudaError((val), __FILE__, __LINE__)

void checkCudaError(cudaError_t result, const char *file, int line);


/// Wrapper to handle allocated Tensor memory and their sizes
struct Gpu_Element {
    float* ptr;
    std::vector<int> sizes;
};


class CudaManager {
    std::vector<cudaStream_t> stream_pool;
    cublasHandle_t cublas_handle;
    mutable std::mutex cuda_mutex;

    static int _get_total_elements(const Gpu_Element& element) ;
    const Gpu_Element& _get_element_no_lock(const std::string& name);
    void _copyHostToDevice_no_lock(const std::string& name, const float* h_ptr);

public:

    ////////////////////////////////////////////////////////////////////////////////
    ///                      Lifecycle & Infrastructure                           //
    ////////////////////////////////////////////////////////////////////////////////

    CudaManager();
    ~CudaManager();

    cudaStream_t get_stream_from_pool() const;
    void synchronize_stream(cudaStream_t stream) const;


    ////////////////////////////////////////////////////////////////////////////////
    ///                          Memory Management                                //
    ////////////////////////////////////////////////////////////////////////////////

    std::map<std::string, Gpu_Element> memory;

    void allocate(const std::string& name, const std::vector<int>& sizes);
    bool exists(const std::string& name) const;
    void free(const std::string& name);

    void copyHostToDevice(const std::string& name, const float* h_ptr);
    std::vector<float> retrieve(const std::string& name);

    float* get(const std::string& name) const { return memory.at(name).ptr; }
    const Gpu_Element& get_element(const std::string& name);


    ////////////////////////////////////////////////////////////////////////////////
    ///                    Basic Math & Element-wise Ops                          //
    ////////////////////////////////////////////////////////////////////////////////

    void zero_grad(const std::string& name, cudaStream_t stream);
    void pad2d(const std::string& name_in, const std::string& name_out, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream);
    void pad2d_backward(const std::string& name_grad_out, const std::string& name_grad_in, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream);
    void init_random_uniform(const std::string& name, float scale, int seed);

    void add_scalar(const std::string& name_in, const std::string& name_out, float scalar, cudaStream_t stream);

    void fill(const std::string& name, float value, cudaStream_t stream);
    void copy(const std::string& name_src, const std::string& name_dest, cudaStream_t stream);

    void add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream);
    void add_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream);
    void subtract(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream);
    void subtract_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream);
    void multiply(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream);
    void divide(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream);
    void exp_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream);

    void multiply_backward(const std::string& grad_out_name, const std::string& other_input_name, const std::string& grad_in_name, cudaStream_t stream);
    void divide_backward(const std::string& A_name, const std::string& B_name, const std::string& grad_out_name, const std::string& grad_A_name, const std::string& grad_B_name, cudaStream_t stream);
    void exp_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream);
    void log_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void log_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, cudaStream_t stream);

    void abs_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void abs_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, cudaStream_t stream);

    void sqrt_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void sqrt_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream);

    void pow_elementwise(const std::string& name_in, const std::string& name_out, float exponent, cudaStream_t stream);
    void pow_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, float exponent, cudaStream_t stream);

    void clamp_elementwise(const std::string& name_in, const std::string& name_out, float min_val, float max_val, cudaStream_t stream);
    void clamp_backward(const std::string& name_grad_out, const std::string& name_in_data, const std::string& name_grad_in, float min_val, float max_val, cudaStream_t stream);

    ////////////////////////////////////////////////////////////////////////////////
    ///                          Matrix Operations                                //
    ////////////////////////////////////////////////////////////////////////////////

    void matmul(const std::string& name_A, const std::string& name_B, const std::string& name_C,
                    bool transpose_A, bool transpose_B,
                    float alpha, float beta,
                    bool use_tensor_cores = false,
                    cudaStream_t stream = nullptr);

    void transpose(const std::string& name_in, const std::string& name_out, cudaStream_t stream);

    void broadcast_add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream);


    ////////////////////////////////////////////////////////////////////////////////
    ///                         Activation Functions                              //
    ////////////////////////////////////////////////////////////////////////////////

    void relu(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void relu_backward(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);

    void sigmoid(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void sigmoid_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);

    void tanh(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void tanh_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);

    void gelu_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void gelu_backward(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);

    void softmax_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void softmax_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream);


    ////////////////////////////////////////////////////////////////////////////////
    ///                            Loss Functions                                 //
    ////////////////////////////////////////////////////////////////////////////////

    void mse_loss_forward(const std::string& pred_name, const std::string& target_name, const std::string& out_name, cudaStream_t stream);
    void mse_loss_backward(const std::string& pred_name, const std::string& target_name, const std::string& grad_in_name, const std::string& grad_out_name, cudaStream_t stream);

    void bce_loss_forward(const std::string& name_pred, const std::string& name_target, const std::string& name_out, cudaStream_t stream);
    void bce_loss_backward(const std::string& name_grad_out, const std::string& name_pred, const std::string& name_target, const std::string& name_grad_in, cudaStream_t stream);


    ////////////////////////////////////////////////////////////////////////////////
    ///                             Optimizers                                    //
    ////////////////////////////////////////////////////////////////////////////////

    void sgd_update(const std::string& data_name, const std::string& grad_name, float learning_rate, cudaStream_t stream);

    void adam_update(
        const std::string& data_name,
        const std::string& grad_name,
        const std::string& m_name,
        const std::string& v_name,
        float learning_rate,
        float beta1,
        float beta2,
        float eps,
        int step,
        float weight_decay,
        cudaStream_t stream);

    void clip_grad_value(const std::string& buffer_name, float clip_value, cudaStream_t stream);


    ////////////////////////////////////////////////////////////////////////////////
    ///                    Advanced Layers & Fused Ops                            //
    ////////////////////////////////////////////////////////////////////////////////

    void matmul_bias_relu_forward(
        const std::string& name_X,
        const std::string& name_W,
        const std::string& name_B,
        const std::string& name_Relu_Out,
        const std::string& name_PreRelu_Out,
        cudaStream_t stream
    );

    void layer_norm_forward(
        const std::string& name_in, const std::string& name_gamma, const std::string& name_beta,
        const std::string& name_out, const std::string& name_mean, const std::string& name_rstd,
        float epsilon, cudaStream_t stream);

    void layer_norm_backward(
        const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma,
        const std::string& name_mean, const std::string& name_rstd, const std::string& name_grad_in,
        const std::string& name_grad_gamma, const std::string& name_grad_beta, cudaStream_t stream);

    void conv2d_forward(const std::string& name_in, const std::string& name_kernel, const std::string& name_out, cudaStream_t stream);
    void conv2d_backward_input(const std::string& name_grad_out, const std::string& name_kernel, const std::string& name_grad_in, cudaStream_t stream);
    void conv2d_backward_kernel(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_kernel, cudaStream_t stream);

    void conv2d_multi_forward(const std::string& name_in, const std::string& name_weight, const std::string& name_bias, const std::string& name_out, int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream);
    void conv2d_multi_backward_input(const std::string& name_grad_out, const std::string& name_weight, const std::string& name_grad_in, int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream);
    void conv2d_multi_backward_weight(const std::string& name_input, const std::string& name_grad_out, const std::string& name_grad_weight, const std::string& name_grad_bias, int C_in, int C_out, int k_h, int k_w, int pad_t, int pad_l, int stride_h, int stride_w, cudaStream_t stream);

    void im2col(const std::string& name_in, const std::string& name_out_col, int kernelHeight, int kernelWidth, cudaStream_t stream);
    void col2im(const std::string& name_col_grad, const std::string& name_in_grad, int kernelHeight, int kernelWidth, cudaStream_t stream);

    void max_pool_1d_forward(const std::string& name_in, const std::string& name_out, const std::string& name_indices, int pool_size, int stride, cudaStream_t stream);
    void max_pool_1d_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_in, cudaStream_t stream);
    void avg_pool_2d_forward(const std::string& name_in, const std::string& name_out, int pool_size, int stride, cudaStream_t stream);
    void avg_pool_2d_backward(const std::string& name_grad_out, const std::string& name_grad_in, int pool_size, int stride, cudaStream_t stream);

    void global_avg_pool_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void global_avg_pool_backward(const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);

    void max_pool_2d_forward(const std::string& name_in, const std::string& name_out, const std::string& name_indices, int pool_size, int stride, cudaStream_t stream);
    void max_pool_2d_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_in, cudaStream_t stream);

    void batch_norm_1d_forward(const std::string& name_in, const std::string& name_gamma, const std::string& name_beta, const std::string& name_running_mean, const std::string& name_running_var, const std::string& name_out, const std::string& name_saved_mean, const std::string& name_saved_inv_var, float momentum, float epsilon, bool is_training, cudaStream_t stream);

    void batch_norm_1d_backward(const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma, const std::string& name_saved_mean, const std::string& name_saved_inv_var, const std::string& name_grad_in, const std::string& name_grad_gamma, const std::string& name_grad_beta, cudaStream_t stream);

    void batch_norm_2d_forward(const std::string& name_in, const std::string& name_gamma, const std::string& name_beta, const std::string& name_running_mean, const std::string& name_running_var, const std::string& name_out, const std::string& name_saved_mean, const std::string& name_saved_inv_var, float momentum, float epsilon, bool is_training, cudaStream_t stream);

    void batch_norm_2d_backward(const std::string& name_grad_out, const std::string& name_in, const std::string& name_gamma, const std::string& name_saved_mean, const std::string& name_saved_inv_var, const std::string& name_grad_in, const std::string& name_grad_gamma, const std::string& name_grad_beta, cudaStream_t stream);


    void scale_matrix(const std::string& name_in, const std::string& name_out, float scale, cudaStream_t stream);
    void scale_matrix_backward(const std::string& name_grad_out, const std::string& name_grad_in, float scale, cudaStream_t stream);


    void dropout_forward(const std::string& name_in, const std::string& name_out, const std::string& name_mask, float drop_rate, int seed, cudaStream_t stream);
    void dropout_backward(const std::string& name_grad_out, const std::string& name_mask, const std::string& name_grad_in, cudaStream_t stream);

    void markov_count(const std::string& name_sequence, const std::string& name_count_table, int order, int num_states, cudaStream_t stream);
    void markov_normalize(const std::string& name_count_table, const std::string& name_prob_table, int num_histories, int num_states, cudaStream_t stream);
    void markov_predict(const std::string& name_history, const std::string& name_prob_table, const std::string& name_out_probs, int order, int num_states, cudaStream_t stream);

    ////////////////////////////////////////////////////////////////////////////////
    ///                        Tensor Manipulation                                //
    ////////////////////////////////////////////////////////////////////////////////

    void slice_row(const std::string& name_in, const std::string& name_out, int row_index, cudaStream_t stream);
    void slice_row_backward(const std::string& name_grad_out, const std::string& name_grad_in, int row_index, cudaStream_t stream);

    void slice_column(const std::string& name_in, const std::string& name_out, int start_col, int end_col, cudaStream_t stream);
    void slice_column_backward(const std::string& name_grad_out, const std::string& name_grad_in, int start_col, int end_col, cudaStream_t stream);

    void stack_rows(const std::vector<std::string>& names_in, const std::string& name_out, int axis, cudaStream_t stream);
    void stack_rows_backward(const std::string& name_grad_out, const std::vector<std::string>& names_grad_in, int axis, cudaStream_t stream);

    void concatenate(const std::string& name_A, const std::string& name_B, const std::string& name_out, int axis, cudaStream_t stream);
    void concatenate_backward(const std::string& name_grad_out, const std::string& name_grad_in_A, const std::string& name_grad_in_B, int axis, int split_index, cudaStream_t stream);


    ////////////////////////////////////////////////////////////////////////////////
    ///                            Reduction Ops                                  //
    ////////////////////////////////////////////////////////////////////////////////

    void sum_reduce(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void sum_reduce_backward(const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream);
    void sum_reduce_columns(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void sum_reduce_rows(const std::string& name_in, const std::string& name_out, cudaStream_t stream);
    void embedding_forward(const std::string& name_indices, const std::string& name_weight, const std::string& name_out, cudaStream_t stream);
    void embedding_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_weight, cudaStream_t stream);


};

#endif