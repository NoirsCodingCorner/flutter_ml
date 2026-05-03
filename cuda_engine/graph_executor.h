#ifndef GRAPH_EXECUTOR_H
#define GRAPH_EXECUTOR_H

#include "manager.h"
#include <cstdint>
#include <vector>
#include <string>

// The strict dictionary of operations. Dart must use these exact integers in the tape.
// Structure: 100-block intervals per category.
// Parity Rule: Even = Forward/Execute, Odd = Backward/Gradient.
enum OpCode : int32_t {

    // 0 - 99: Data & Memory Management
    OP_LOAD_SAMPLE = 2,
    OP_STORE_SAMPLE = 4,
    OP_COPY = 6,
    OP_FILL = 8,
    OP_ZERO_GRAD = 10,

    // 100 - 199: Basic Math (Scalars & Element-wise)
    OP_ADD = 100,
    OP_ADD_INTO = 102,
    OP_ADD_SCALAR = 104,
    OP_SUBTRACT = 106,
    OP_SUBTRACT_INTO = 108,
    OP_MULTIPLY = 110,
    OP_MULTIPLY_BACKWARD = 111,
    OP_DIVIDE = 112,
    OP_DIVIDE_BACKWARD = 113,
    OP_EXP_ELEMENTWISE = 114,
    OP_EXP_BACKWARD = 115,
    OP_LOG_ELEMENTWISE = 116,
    OP_LOG_BACKWARD = 117,
    OP_ABS_ELEMENTWISE = 118,
    OP_ABS_BACKWARD = 119,
    OP_SQRT_ELEMENTWISE = 120,
    OP_SQRT_BACKWARD = 121,
    OP_POW_ELEMENTWISE = 122,
    OP_POW_BACKWARD = 123,
    OP_CLAMP_ELEMENTWISE = 124,
    OP_CLAMP_BACKWARD = 125,

    // 200 - 299: Matrix Operations
    OP_MATMUL = 200,
    OP_TRANSPOSE = 202,
    OP_BROADCAST_ADD = 204,
    OP_SCALE_MATRIX = 206,
    OP_SCALE_MATRIX_BACKWARD = 207,

    // 300 - 399: Activations
    OP_RELU = 300,
    OP_RELU_BACKWARD = 301,
    OP_SIGMOID = 302,
    OP_SIGMOID_BACKWARD = 303,
    OP_TANH = 304,
    OP_TANH_BACKWARD = 305,
    OP_GELU_FORWARD = 306,
    OP_GELU_BACKWARD = 307,
    OP_SOFTMAX_FORWARD = 308,
    OP_SOFTMAX_BACKWARD = 309,

    // 400 - 499: Loss Functions
    OP_MSE_LOSS_FORWARD = 400,
    OP_MSE_LOSS_BACKWARD = 401,
    OP_BCE_LOSS_FORWARD = 402,
    OP_BCE_LOSS_BACKWARD = 403,

    // 500 - 599: Optimizers
    OP_SGD_UPDATE = 500,
    OP_ADAM_UPDATE = 502,
    OP_CLIP_GRAD_VALUE = 504,

    // 600 - 699: Reductions
    OP_SUM_REDUCE = 600,
    OP_SUM_REDUCE_BACKWARD = 601,
    OP_SUM_REDUCE_COLUMNS = 602,
    OP_SUM_REDUCE_ROWS = 604,
    OP_EMBEDDING_FORWARD = 606,
    OP_EMBEDDING_BACKWARD = 607,

    // 700 - 799: Tensor Manipulation
    OP_SLICE_ROW = 700,
    OP_SLICE_ROW_BACKWARD = 701,
    OP_SLICE_COLUMN = 702,
    OP_SLICE_COLUMN_BACKWARD = 703,
    OP_STACK_ROWS = 704,
    OP_STACK_ROWS_BACKWARD = 705,
    OP_CONCATENATE = 706,
    OP_CONCATENATE_BACKWARD = 707,
    OP_PAD2D = 708,
    OP_PAD2D_BACKWARD = 709,

    // 800 - 999: Advanced Spatial & Sequence Layers
    OP_CONV2D_FORWARD = 800,
    OP_CONV2D_BACKWARD_INPUT = 801,
    OP_CONV2D_BACKWARD_KERNEL = 803,
    OP_CONV2D_MULTI_FORWARD = 804,
    OP_CONV2D_MULTI_BACKWARD_INPUT = 805,
    OP_CONV2D_MULTI_BACKWARD_WEIGHT = 807,
    OP_IM2COL = 808,
    OP_COL2IM = 809,
    OP_MAX_POOL_1D_FORWARD = 810,
    OP_MAX_POOL_1D_BACKWARD = 811,
    OP_MAX_POOL_2D_FORWARD = 812,
    OP_MAX_POOL_2D_BACKWARD = 813,
    OP_AVG_POOL_2D_FORWARD = 814,
    OP_AVG_POOL_2D_BACKWARD = 815,
    OP_GLOBAL_AVG_POOL_FORWARD = 816,
    OP_GLOBAL_AVG_POOL_BACKWARD = 817,
    OP_BATCH_NORM_1D_FORWARD = 820,
    OP_BATCH_NORM_1D_BACKWARD = 821,
    OP_BATCH_NORM_2D_FORWARD = 822,
    OP_BATCH_NORM_2D_BACKWARD = 823,
    OP_LAYER_NORM_FORWARD = 824,
    OP_LAYER_NORM_BACKWARD = 825,
    OP_DROPOUT_FORWARD = 826,
    OP_DROPOUT_BACKWARD = 827,

    OP_MARKOV_COUNT = 900,
    OP_MARKOV_NORMALIZE = 902,
    OP_MARKOV_PREDICT = 904,

    // 1000+: Fused Kernels
    // Follows the Inheritance Rule: 1000 + Base ID (MATMUL = 200 -> 1200)
    OP_MATMUL_BIAS_RELU_FORWARD = 1200
};

class GraphExecutor {
private:
    CudaManager& manager_;
    bool debug_;

public:
    explicit GraphExecutor(CudaManager& manager, bool debug = false)
        : manager_(manager), debug_(debug) {}
    void run_tape(const uint8_t* tape, int total_bytes, bool sync = true) const;
};

#endif // GRAPH_EXECUTOR_H