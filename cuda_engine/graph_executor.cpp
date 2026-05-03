#include "graph_executor.h"
#include <stdexcept>
#include <vector>
#include <string>

////////////////////////////////////////////////////////////////////////////////
///                        Byte Decoder Utilities                             //
////////////////////////////////////////////////////////////////////////////////
template<typename T>
inline T read_tape(const uint8_t* tape, int& offset) {
    T value = *reinterpret_cast<const T*>(tape + offset);
    offset += sizeof(T);
    return value;
}

/// Reads a length-prefixed string: [2-byte uint16_t length] [chars...]
inline std::string read_string(const uint8_t* tape, int& offset) {

    uint16_t len = *reinterpret_cast<const uint16_t*>(tape + offset);
    offset += sizeof(uint16_t);

    std::string str(reinterpret_cast<const char*>(tape + offset), len);
    offset += len;

    return str;
}


////////////////////////////////////////////////////////////////////////////////
///                            Execution Loop                                 //
////////////////////////////////////////////////////////////////////////////////

void GraphExecutor::run_tape(const uint8_t* tape, int total_bytes, bool sync) const {
    int offset = 0;
    cudaStream_t stream = manager_.get_stream_from_pool();

    while (offset < total_bytes) {
        auto op = read_tape<int32_t>(tape, offset);

        switch (static_cast<OpCode>(op)) {

            // --- 0 - 99: Data & Memory Management ---
            case OP_LOAD_SAMPLE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t sample_idx   = read_tape<int32_t>(tape, offset);
                manager_.slice_row(name_in, name_out, sample_idx, stream);
                break;
            }
            case OP_STORE_SAMPLE: {
                std::string name_in   = read_string(tape, offset);
                std::string name_dest = read_string(tape, offset);
                int32_t sample_idx    = read_tape<int32_t>(tape, offset);
                manager_.slice_row_backward(name_in, name_dest, sample_idx, stream);
                break;
            }
            case OP_COPY: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.copy(name_in, name_out, stream);
                break;
            }
            case OP_FILL: {
                std::string name_out = read_string(tape, offset);
                float value          = read_tape<float>(tape, offset);
                manager_.fill(name_out, value, stream);
                break;
            }
            case OP_ZERO_GRAD: {
                std::string name = read_string(tape, offset);
                manager_.zero_grad(name, stream);
                break;
            }

            // --- 100 - 199: Basic Math (Scalars & Element-wise) ---
            case OP_ADD: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                manager_.add(name_A, name_B, name_C, stream);
                break;
            }
            case OP_ADD_INTO: {
                std::string name_src  = read_string(tape, offset);
                std::string name_dest = read_string(tape, offset);
                manager_.add_into(name_src, name_dest, stream);
                break;
            }
            case OP_ADD_SCALAR: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                float scalar         = read_tape<float>(tape, offset);
                manager_.add_scalar(name_in, name_out, scalar, stream);
                break;
            }
            case OP_SUBTRACT: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                manager_.subtract(name_A, name_B, name_C, stream);
                break;
            }
            case OP_SUBTRACT_INTO: {
                std::string name_src  = read_string(tape, offset);
                std::string name_dest = read_string(tape, offset);
                manager_.subtract_into(name_src, name_dest, stream);
                break;
            }
            case OP_MULTIPLY: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                manager_.multiply(name_A, name_B, name_C, stream);
                break;
            }
            case OP_MULTIPLY_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_other_in = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.multiply_backward(name_grad_out, name_other_in, name_grad_in, stream);
                break;
            }
            case OP_DIVIDE: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                manager_.divide(name_A, name_B, name_C, stream);
                break;
            }
            case OP_DIVIDE_BACKWARD: {
                std::string name_A        = read_string(tape, offset);
                std::string name_B        = read_string(tape, offset);
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_A   = read_string(tape, offset);
                std::string name_grad_B   = read_string(tape, offset);
                manager_.divide_backward(name_A, name_B, name_grad_out, name_grad_A, name_grad_B, stream);
                break;
            }
            case OP_EXP_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.exp_elementwise(name_in, name_out, stream);
                break;
            }
            case OP_EXP_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_out_data = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.exp_backward(name_grad_out, name_out_data, name_grad_in, stream);
                break;
            }
            case OP_LOG_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.log_elementwise(name_in, name_out, stream);
                break;
            }
            case OP_LOG_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_in_data  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.log_backward(name_grad_out, name_in_data, name_grad_in, stream);
                break;
            }
            case OP_ABS_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.abs_elementwise(name_in, name_out, stream);
                break;
            }
            case OP_ABS_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_in_data  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.abs_backward(name_grad_out, name_in_data, name_grad_in, stream);
                break;
            }
            case OP_SQRT_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.sqrt_elementwise(name_in, name_out, stream);
                break;
            }
            case OP_SQRT_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_out_data = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.sqrt_backward(name_grad_out, name_out_data, name_grad_in, stream);
                break;
            }
            case OP_POW_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                float exponent       = read_tape<float>(tape, offset);
                manager_.pow_elementwise(name_in, name_out, exponent, stream);
                break;
            }
            case OP_POW_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_in_data  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                float exponent            = read_tape<float>(tape, offset);
                manager_.pow_backward(name_grad_out, name_in_data, name_grad_in, exponent, stream);
                break;
            }
            case OP_CLAMP_ELEMENTWISE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                float min_val        = read_tape<float>(tape, offset);
                float max_val        = read_tape<float>(tape, offset);
                manager_.clamp_elementwise(name_in, name_out, min_val, max_val, stream);
                break;
            }
            case OP_CLAMP_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_in_data  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                float min_val             = read_tape<float>(tape, offset);
                float max_val             = read_tape<float>(tape, offset);
                manager_.clamp_backward(name_grad_out, name_in_data, name_grad_in, min_val, max_val, stream);
                break;
            }

            // --- 200 - 299: Matrix Operations ---
            case OP_MATMUL: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                bool trans_A       = read_tape<bool>(tape, offset);
                bool trans_B       = read_tape<bool>(tape, offset);
                float alpha        = read_tape<float>(tape, offset);
                float beta         = read_tape<float>(tape, offset);
                bool t_cores       = read_tape<bool>(tape, offset);
                manager_.matmul(name_A, name_B, name_C, trans_A, trans_B, alpha, beta, t_cores, stream);
                break;
            }
            case OP_TRANSPOSE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.transpose(name_in, name_out, stream);
                break;
            }
            case OP_BROADCAST_ADD: {
                std::string name_A = read_string(tape, offset);
                std::string name_B = read_string(tape, offset);
                std::string name_C = read_string(tape, offset);
                manager_.broadcast_add(name_A, name_B, name_C, stream);
                break;
            }
            case OP_SCALE_MATRIX: {
                std::string in_name = read_string(tape, offset);
                std::string out_name = read_string(tape, offset);
                float scale = read_tape<float>(tape, offset);
                manager_.scale_matrix(in_name, out_name, scale, stream);
                break;
            }
            case OP_SCALE_MATRIX_BACKWARD: {
                std::string grad_out_name = read_string(tape, offset);
                std::string grad_in_name = read_string(tape, offset);
                float scale = read_tape<float>(tape, offset);
                manager_.scale_matrix_backward(grad_out_name, grad_in_name, scale, stream);
                break;
            }

            // --- 300 - 399: Activations ---
            case OP_RELU: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.relu(name_in, name_out, stream);
                break;
            }
            case OP_RELU_BACKWARD: {
                std::string name_in       = read_string(tape, offset);
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.relu_backward(name_in, name_grad_out, name_grad_in, stream);
                break;
            }
            case OP_SIGMOID: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.sigmoid(name_in, name_out, stream);
                break;
            }
            case OP_SIGMOID_BACKWARD: {
                std::string name_out_data = read_string(tape, offset);
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.sigmoid_backward(name_out_data, name_grad_out, name_grad_in, stream);
                break;
            }
            case OP_TANH: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.tanh(name_in, name_out, stream);
                break;
            }
            case OP_TANH_BACKWARD: {
                std::string name_out_data = read_string(tape, offset);
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.tanh_backward(name_out_data, name_grad_out, name_grad_in, stream);
                break;
            }
            case OP_GELU_FORWARD: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.gelu_forward(name_in, name_out, stream);
                break;
            }
            case OP_GELU_BACKWARD: {
                std::string name_in       = read_string(tape, offset);
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.gelu_backward(name_in, name_grad_out, name_grad_in, stream);
                break;
            }
            case OP_SOFTMAX_FORWARD: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.softmax_forward(name_in, name_out, stream);
                break;
            }
            case OP_SOFTMAX_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_out_data = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.softmax_backward(name_grad_out, name_out_data, name_grad_in, stream);
                break;
            }

            // --- 400 - 499: Loss Functions ---
            case OP_MSE_LOSS_FORWARD: {
                std::string name_pred   = read_string(tape, offset);
                std::string name_target = read_string(tape, offset);
                std::string name_out    = read_string(tape, offset);
                manager_.mse_loss_forward(name_pred, name_target, name_out, stream);
                break;
            }
            case OP_MSE_LOSS_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_pred     = read_string(tape, offset);
                std::string name_target   = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.mse_loss_backward(name_pred, name_target, name_grad_in, name_grad_out, stream);
                break;
            }
            case OP_BCE_LOSS_FORWARD: {
                std::string name_pred   = read_string(tape, offset);
                std::string name_target = read_string(tape, offset);
                std::string name_out    = read_string(tape, offset);
                manager_.bce_loss_forward(name_pred, name_target, name_out, stream);
                break;
            }
            case OP_BCE_LOSS_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_pred     = read_string(tape, offset);
                std::string name_target   = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.bce_loss_backward(name_grad_out, name_pred, name_target, name_grad_in, stream);
                break;
            }

            // --- 500 - 599: Optimizers ---
            case OP_SGD_UPDATE: {
                std::string name_data = read_string(tape, offset);
                std::string name_grad = read_string(tape, offset);
                float lr              = read_tape<float>(tape, offset);
                manager_.sgd_update(name_data, name_grad, lr, stream);
                break;
            }
            case OP_ADAM_UPDATE: {
                std::string name_data = read_string(tape, offset);
                std::string name_grad = read_string(tape, offset);
                std::string name_m    = read_string(tape, offset);
                std::string name_v    = read_string(tape, offset);
                float lr              = read_tape<float>(tape, offset);
                float b1              = read_tape<float>(tape, offset);
                float b2              = read_tape<float>(tape, offset);
                float eps             = read_tape<float>(tape, offset);
                int32_t step          = read_tape<int32_t>(tape, offset);
                float wd              = read_tape<float>(tape, offset);
                manager_.adam_update(name_data, name_grad, name_m, name_v, lr, b1, b2, eps, step, wd, stream);
                break;
            }
            case OP_CLIP_GRAD_VALUE: {
                std::string name_buffer = read_string(tape, offset);
                float clip_val          = read_tape<float>(tape, offset);
                manager_.clip_grad_value(name_buffer, clip_val, stream);
                break;
            }

            // --- 600 - 699: Reductions ---
            case OP_SUM_REDUCE: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.sum_reduce(name_in, name_out, stream);
                break;
            }
            case OP_SUM_REDUCE_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.sum_reduce_backward(name_grad_out, name_grad_in, stream);
                break;
            }
            case OP_SUM_REDUCE_COLUMNS: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.sum_reduce_columns(name_in, name_out, stream);
                break;
            }
            case OP_SUM_REDUCE_ROWS: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                manager_.sum_reduce_rows(name_in, name_out, stream);
                break;
            }
            case OP_EMBEDDING_FORWARD: {
                std::string name_indices = read_string(tape, offset);
                std::string name_weight  = read_string(tape, offset);
                std::string name_out     = read_string(tape, offset);
                manager_.embedding_forward(name_indices, name_weight, name_out, stream);
                break;
            }
            case OP_EMBEDDING_BACKWARD: {
                std::string name_grad_out    = read_string(tape, offset);
                std::string name_indices     = read_string(tape, offset);
                std::string name_grad_weight = read_string(tape, offset);
                manager_.embedding_backward(name_grad_out, name_indices, name_grad_weight, stream);
                break;
            }

            // --- 700 - 799: Tensor Manipulation ---
            case OP_SLICE_ROW: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t row          = read_tape<int32_t>(tape, offset);
                manager_.slice_row(name_in, name_out, row, stream);
                break;
            }
            case OP_SLICE_ROW_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                int32_t row               = read_tape<int32_t>(tape, offset);
                manager_.slice_row_backward(name_grad_out, name_grad_in, row, stream);
                break;
            }
            case OP_SLICE_COLUMN: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t start        = read_tape<int32_t>(tape, offset);
                int32_t end          = read_tape<int32_t>(tape, offset);
                manager_.slice_column(name_in, name_out, start, end, stream);
                break;
            }
            case OP_SLICE_COLUMN_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                int32_t start             = read_tape<int32_t>(tape, offset);
                int32_t end               = read_tape<int32_t>(tape, offset);
                manager_.slice_column_backward(name_grad_out, name_grad_in, start, end, stream);
                break;
            }
            case OP_STACK_ROWS: {
                int32_t count = read_tape<int32_t>(tape, offset);
                std::vector<std::string> names_in;
                names_in.reserve(count);
                for(int i = 0; i < count; ++i) {
                    names_in.push_back(read_string(tape, offset));
                }
                std::string name_out = read_string(tape, offset);
                int32_t axis         = read_tape<int32_t>(tape, offset);
                manager_.stack_rows(names_in, name_out, axis, stream);
                break;
            }
            case OP_STACK_ROWS_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                int32_t count             = read_tape<int32_t>(tape, offset);
                std::vector<std::string> names_grad_in;
                names_grad_in.reserve(count);
                for(int i = 0; i < count; ++i) {
                    names_grad_in.push_back(read_string(tape, offset));
                }
                int32_t axis = read_tape<int32_t>(tape, offset);
                manager_.stack_rows_backward(name_grad_out, names_grad_in, axis, stream);
                break;
            }
            case OP_CONCATENATE: {
                std::string name_A   = read_string(tape, offset);
                std::string name_B   = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t axis         = read_tape<int32_t>(tape, offset);
                manager_.concatenate(name_A, name_B, name_out, axis, stream);
                break;
            }
            case OP_CONCATENATE_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_inA = read_string(tape, offset);
                std::string name_grad_inB = read_string(tape, offset);
                int32_t axis              = read_tape<int32_t>(tape, offset);
                int32_t split             = read_tape<int32_t>(tape, offset);
                manager_.concatenate_backward(name_grad_out, name_grad_inA, name_grad_inB, axis, split, stream);
                break;
            }
            case OP_PAD2D: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t pad_t        = read_tape<int32_t>(tape, offset);
                int32_t pad_b        = read_tape<int32_t>(tape, offset);
                int32_t pad_l        = read_tape<int32_t>(tape, offset);
                int32_t pad_r        = read_tape<int32_t>(tape, offset);
                manager_.pad2d(name_in, name_out, pad_t, pad_b, pad_l, pad_r, stream);
                break;
            }
            case OP_PAD2D_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                int32_t pad_t             = read_tape<int32_t>(tape, offset);
                int32_t pad_b             = read_tape<int32_t>(tape, offset);
                int32_t pad_l             = read_tape<int32_t>(tape, offset);
                int32_t pad_r             = read_tape<int32_t>(tape, offset);
                manager_.pad2d_backward(name_grad_out, name_grad_in, pad_t, pad_b, pad_l, pad_r, stream);
                break;
            }

            // --- 800 - 999: Advanced Spatial & Sequence Layers ---
            case OP_CONV2D_FORWARD: {
                std::string name_in     = read_string(tape, offset);
                std::string name_kernel = read_string(tape, offset);
                std::string name_out    = read_string(tape, offset);
                manager_.conv2d_forward(name_in, name_kernel, name_out, stream);
                break;
            }
            case OP_CONV2D_BACKWARD_INPUT: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_kernel   = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.conv2d_backward_input(name_grad_out, name_kernel, name_grad_in, stream);
                break;
            }
            case OP_CONV2D_BACKWARD_KERNEL: {
                std::string name_in          = read_string(tape, offset);
                std::string name_grad_out    = read_string(tape, offset);
                std::string name_grad_kernel = read_string(tape, offset);
                manager_.conv2d_backward_kernel(name_in, name_grad_out, name_grad_kernel, stream);
                break;
            }
            case OP_CONV2D_MULTI_FORWARD: {
                std::string name_in    = read_string(tape, offset);
                std::string name_w     = read_string(tape, offset);
                std::string name_b     = read_string(tape, offset);
                std::string name_out   = read_string(tape, offset);
                int32_t c_in  = read_tape<int32_t>(tape, offset);
                int32_t c_out = read_tape<int32_t>(tape, offset);
                int32_t k_h   = read_tape<int32_t>(tape, offset);
                int32_t k_w   = read_tape<int32_t>(tape, offset);
                int32_t pad_t = read_tape<int32_t>(tape, offset);
                int32_t pad_l = read_tape<int32_t>(tape, offset);
                int32_t stride_h = read_tape<int32_t>(tape, offset);
                int32_t stride_w = read_tape<int32_t>(tape, offset);
                manager_.conv2d_multi_forward(name_in, name_w, name_b, name_out, c_in, c_out, k_h, k_w, pad_t, pad_l, stride_h, stride_w, stream);
                break;
            }
            case OP_CONV2D_MULTI_BACKWARD_INPUT: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_weight   = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                int32_t c_in  = read_tape<int32_t>(tape, offset);
                int32_t c_out = read_tape<int32_t>(tape, offset);
                int32_t k_h   = read_tape<int32_t>(tape, offset);
                int32_t k_w   = read_tape<int32_t>(tape, offset);
                int32_t pad_t = read_tape<int32_t>(tape, offset);
                int32_t pad_l = read_tape<int32_t>(tape, offset);
                int32_t stride_h = read_tape<int32_t>(tape, offset);
                int32_t stride_w = read_tape<int32_t>(tape, offset);
                manager_.conv2d_multi_backward_input(name_grad_out, name_weight, name_grad_in, c_in, c_out, k_h, k_w, pad_t, pad_l, stride_h, stride_w, stream);
                break;
            }
            case OP_CONV2D_MULTI_BACKWARD_WEIGHT: {
                std::string name_input       = read_string(tape, offset);
                std::string name_grad_out    = read_string(tape, offset);
                std::string name_grad_weight = read_string(tape, offset);
                std::string name_grad_bias   = read_string(tape, offset);
                int32_t c_in  = read_tape<int32_t>(tape, offset);
                int32_t c_out = read_tape<int32_t>(tape, offset);
                int32_t k_h   = read_tape<int32_t>(tape, offset);
                int32_t k_w   = read_tape<int32_t>(tape, offset);
                int32_t pad_t = read_tape<int32_t>(tape, offset);
                int32_t pad_l = read_tape<int32_t>(tape, offset);
                int32_t stride_h = read_tape<int32_t>(tape, offset);
                int32_t stride_w = read_tape<int32_t>(tape, offset);
                manager_.conv2d_multi_backward_weight(name_input, name_grad_out, name_grad_weight, name_grad_bias, c_in, c_out, k_h, k_w, pad_t, pad_l, stride_h, stride_w, stream);
                break;
            }
            case OP_IM2COL: {
                std::string name_in  = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                int32_t kh           = read_tape<int32_t>(tape, offset);
                int32_t kw           = read_tape<int32_t>(tape, offset);
                manager_.im2col(name_in, name_out, kh, kw, stream);
                break;
            }
            case OP_COL2IM: {
                std::string name_col_grad = read_string(tape, offset);
                std::string name_in_grad  = read_string(tape, offset);
                int32_t kh                = read_tape<int32_t>(tape, offset);
                int32_t kw                = read_tape<int32_t>(tape, offset);
                manager_.col2im(name_col_grad, name_in_grad, kh, kw, stream);
                break;
            }
            case OP_MAX_POOL_1D_FORWARD: {
                std::string name_in      = read_string(tape, offset);
                std::string name_out     = read_string(tape, offset);
                std::string name_indices = read_string(tape, offset);
                int32_t pool_size        = read_tape<int32_t>(tape, offset);
                int32_t stride           = read_tape<int32_t>(tape, offset);
                manager_.max_pool_1d_forward(name_in, name_out, name_indices, pool_size, stride, stream);
                break;
            }
            case OP_MAX_POOL_1D_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_indices  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.max_pool_1d_backward(name_grad_out, name_indices, name_grad_in, stream);
                break;
            }
            case OP_MAX_POOL_2D_FORWARD: {
                std::string name_in      = read_string(tape, offset);
                std::string name_out     = read_string(tape, offset);
                std::string name_indices = read_string(tape, offset);
                int32_t pool_size        = read_tape<int32_t>(tape, offset);
                int32_t stride           = read_tape<int32_t>(tape, offset);
                manager_.max_pool_2d_forward(name_in, name_out, name_indices, pool_size, stride, stream);
                break;
            }
            case OP_MAX_POOL_2D_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_indices  = read_string(tape, offset);
                std::string name_grad_in  = read_string(tape, offset);
                manager_.max_pool_2d_backward(name_grad_out, name_indices, name_grad_in, stream);
                break;
            }
            case OP_AVG_POOL_2D_FORWARD: {
                std::string in_name = read_string(tape, offset);
                std::string out_name = read_string(tape, offset);
                int pool_size = read_tape<int>(tape, offset);
                int stride = read_tape<int>(tape, offset);
                manager_.avg_pool_2d_forward(in_name, out_name, pool_size, stride, stream);
                break;
            }
            case OP_AVG_POOL_2D_BACKWARD: {
                std::string grad_out_name = read_string(tape, offset);
                std::string grad_in_name = read_string(tape, offset);
                int pool_size = read_tape<int>(tape, offset);
                int stride = read_tape<int>(tape, offset);
                manager_.avg_pool_2d_backward(grad_out_name, grad_in_name, pool_size, stride, stream);
                break;
            }
            case OP_GLOBAL_AVG_POOL_FORWARD: {
                std::string in_name = read_string(tape, offset);
                std::string out_name = read_string(tape, offset);
                manager_.global_avg_pool_forward(in_name, out_name, stream);
                break;
            }
            case OP_GLOBAL_AVG_POOL_BACKWARD: {
                std::string grad_out_name = read_string(tape, offset);
                std::string grad_in_name = read_string(tape, offset);
                manager_.global_avg_pool_backward(grad_out_name, grad_in_name, stream);
                break;
            }
            case OP_BATCH_NORM_1D_FORWARD: {
                std::string name_in    = read_string(tape, offset);
                std::string name_gamma = read_string(tape, offset);
                std::string name_beta  = read_string(tape, offset);
                std::string name_rm    = read_string(tape, offset);
                std::string name_rv    = read_string(tape, offset);
                std::string name_out   = read_string(tape, offset);
                std::string name_sm    = read_string(tape, offset);
                std::string name_siv   = read_string(tape, offset);
                float momentum         = read_tape<float>(tape, offset);
                float epsilon          = read_tape<float>(tape, offset);
                bool is_training       = read_tape<bool>(tape, offset);
                manager_.batch_norm_1d_forward(name_in, name_gamma, name_beta, name_rm, name_rv, name_out, name_sm, name_siv, momentum, epsilon, is_training, stream);
                break;
            }
            case OP_BATCH_NORM_1D_BACKWARD: {
                std::string name_grad_out   = read_string(tape, offset);
                std::string name_in         = read_string(tape, offset);
                std::string name_gamma      = read_string(tape, offset);
                std::string name_sm         = read_string(tape, offset);
                std::string name_siv        = read_string(tape, offset);
                std::string name_grad_in    = read_string(tape, offset);
                std::string name_grad_gamma = read_string(tape, offset);
                std::string name_grad_beta  = read_string(tape, offset);
                manager_.batch_norm_1d_backward(name_grad_out, name_in, name_gamma, name_sm, name_siv, name_grad_in, name_grad_gamma, name_grad_beta, stream);
                break;
            }
            case OP_BATCH_NORM_2D_FORWARD: {
                std::string name_in    = read_string(tape, offset);
                std::string name_gamma = read_string(tape, offset);
                std::string name_beta  = read_string(tape, offset);
                std::string name_rm    = read_string(tape, offset);
                std::string name_rv    = read_string(tape, offset);
                std::string name_out   = read_string(tape, offset);
                std::string name_sm    = read_string(tape, offset);
                std::string name_siv   = read_string(tape, offset);
                float momentum         = read_tape<float>(tape, offset);
                float epsilon          = read_tape<float>(tape, offset);
                bool is_training       = read_tape<bool>(tape, offset);
                manager_.batch_norm_2d_forward(name_in, name_gamma, name_beta, name_rm, name_rv, name_out, name_sm, name_siv, momentum, epsilon, is_training, stream);
                break;
            }
            case OP_BATCH_NORM_2D_BACKWARD: {
                std::string name_grad_out   = read_string(tape, offset);
                std::string name_in         = read_string(tape, offset);
                std::string name_gamma      = read_string(tape, offset);
                std::string name_sm         = read_string(tape, offset);
                std::string name_siv        = read_string(tape, offset);
                std::string name_grad_in    = read_string(tape, offset);
                std::string name_grad_gamma = read_string(tape, offset);
                std::string name_grad_beta  = read_string(tape, offset);
                manager_.batch_norm_2d_backward(name_grad_out, name_in, name_gamma, name_sm, name_siv, name_grad_in, name_grad_gamma, name_grad_beta, stream);
                break;
            }
            case OP_LAYER_NORM_FORWARD: {
                std::string name_in    = read_string(tape, offset);
                std::string name_gamma = read_string(tape, offset);
                std::string name_beta  = read_string(tape, offset);
                std::string name_out   = read_string(tape, offset);
                std::string name_mean  = read_string(tape, offset);
                std::string name_rstd  = read_string(tape, offset);
                float eps              = read_tape<float>(tape, offset);
                manager_.layer_norm_forward(name_in, name_gamma, name_beta, name_out, name_mean, name_rstd, eps, stream);
                break;
            }
            case OP_LAYER_NORM_BACKWARD: {
                std::string name_grad_out  = read_string(tape, offset);
                std::string name_in        = read_string(tape, offset);
                std::string name_gamma     = read_string(tape, offset);
                std::string name_mean      = read_string(tape, offset);
                std::string name_rstd      = read_string(tape, offset);
                std::string name_grad_in   = read_string(tape, offset);
                std::string name_grad_gamma= read_string(tape, offset);
                std::string name_grad_beta = read_string(tape, offset);
                manager_.layer_norm_backward(name_grad_out, name_in, name_gamma, name_mean, name_rstd, name_grad_in, name_grad_gamma, name_grad_beta, stream);
                break;
            }
            case OP_DROPOUT_FORWARD: {
                std::string name_in = read_string(tape, offset);
                std::string name_out = read_string(tape, offset);
                std::string name_mask = read_string(tape, offset);
                float drop_rate = read_tape<float>(tape, offset);
                int32_t seed = read_tape<int32_t>(tape, offset);
                manager_.dropout_forward(name_in, name_out, name_mask, drop_rate, seed, stream);
                break;
            }
            case OP_DROPOUT_BACKWARD: {
                std::string name_grad_out = read_string(tape, offset);
                std::string name_mask = read_string(tape, offset);
                std::string name_grad_in = read_string(tape, offset);
                manager_.dropout_backward(name_grad_out, name_mask, name_grad_in, stream);
                break;
            }
            case OP_MARKOV_COUNT: {
                std::string name_seq   = read_string(tape, offset);
                std::string name_count = read_string(tape, offset);
                int32_t order          = read_tape<int32_t>(tape, offset);
                int32_t num_states     = read_tape<int32_t>(tape, offset);
                manager_.markov_count(name_seq, name_count, order, num_states, stream);
                break;
            }
            case OP_MARKOV_NORMALIZE: {
                std::string name_count = read_string(tape, offset);
                std::string name_prob  = read_string(tape, offset);
                int32_t num_histories  = read_tape<int32_t>(tape, offset);
                int32_t num_states     = read_tape<int32_t>(tape, offset);
                manager_.markov_normalize(name_count, name_prob, num_histories, num_states, stream);
                break;
            }
            case OP_MARKOV_PREDICT: {
                std::string name_hist  = read_string(tape, offset);
                std::string name_prob  = read_string(tape, offset);
                std::string name_out   = read_string(tape, offset);
                int32_t order          = read_tape<int32_t>(tape, offset);
                int32_t num_states     = read_tape<int32_t>(tape, offset);
                manager_.markov_predict(name_hist, name_prob, name_out, order, num_states, stream);
                break;
            }

            // --- 1000+: Fused Kernels ---
            case OP_MATMUL_BIAS_RELU_FORWARD: {
                std::string name_X         = read_string(tape, offset);
                std::string name_W         = read_string(tape, offset);
                std::string name_B         = read_string(tape, offset);
                std::string name_Relu_Out  = read_string(tape, offset);
                std::string name_PreR_Out  = read_string(tape, offset);
                manager_.matmul_bias_relu_forward(name_X, name_W, name_B, name_Relu_Out, name_PreR_Out, stream);
                break;
            }

            default:
                throw std::runtime_error("GraphExecutor: Unknown OpCode encountered in tape.");
        }
    }

    if (sync) {
        manager_.synchronize_stream(stream);
    }
}