/// OpCodes are used as the communication commands with the native implementations.
/// Most OpCodes are responsible for calling a function/calculation on its provided arguments.
library;


// =========================================================================
// C++ Engine Instruction Map (Byte Tape OpCodes)
// Ensure these perfectly match the C++ OpCode enum in the native implementations.
// If not, please check versions of the current dart and cuda engine
// =========================================================================


// --- 0 - 99: Data & Memory Management ---
const int OP_LOAD_SAMPLE              = 2;
const int OP_STORE_SAMPLE             = 4;
const int OP_COPY                     = 6;
const int OP_FILL                     = 8;
const int OP_ZERO_GRAD                = 10;

// --- 100 - 199: Basic Math (Scalars & Element-wise) ---
const int OP_ADD                      = 100;
const int OP_ADD_INTO                 = 102;
const int OP_ADD_SCALAR               = 104;
const int OP_SUBTRACT                 = 106;
const int OP_SUBTRACT_INTO            = 108;
const int OP_MULTIPLY                 = 110;
const int OP_MULTIPLY_BACKWARD        = 111;
const int OP_DIVIDE                   = 112;
const int OP_DIVIDE_BACKWARD          = 113;
const int OP_EXP_ELEMENTWISE          = 114;
const int OP_EXP_BACKWARD             = 115;
const int OP_LOG_ELEMENTWISE          = 116;
const int OP_LOG_BACKWARD             = 117;
const int OP_ABS_ELEMENTWISE          = 118;
const int OP_ABS_BACKWARD             = 119;
const int OP_SQRT_ELEMENTWISE         = 120;
const int OP_SQRT_BACKWARD            = 121;
const int OP_POW_ELEMENTWISE          = 122;
const int OP_POW_BACKWARD             = 123;
const int OP_CLAMP_ELEMENTWISE        = 124;
const int OP_CLAMP_BACKWARD           = 125;

// --- 200 - 299: Matrix Operations ---
const int OP_MATMUL                   = 200;
const int OP_TRANSPOSE                = 202;
const int OP_BROADCAST_ADD            = 204;
const int OP_SCALE_MATRIX             = 206;
const int OP_SCALE_MATRIX_BACKWARD    = 207;

// --- 300 - 399: Activations ---
const int OP_RELU                     = 300;
const int OP_RELU_BACKWARD            = 301;
const int OP_SIGMOID                  = 302;
const int OP_SIGMOID_BACKWARD         = 303;
const int OP_TANH                     = 304;
const int OP_TANH_BACKWARD            = 305;
const int OP_GELU_FORWARD             = 306;
const int OP_GELU_BACKWARD            = 307;
const int OP_SOFTMAX_FORWARD          = 308;
const int OP_SOFTMAX_BACKWARD         = 309;

// --- 400 - 499: Loss Functions ---
const int OP_MSE_LOSS_FORWARD         = 400;
const int OP_MSE_LOSS_BACKWARD        = 401;
const int OP_BCE_LOSS_FORWARD         = 402;
const int OP_BCE_LOSS_BACKWARD        = 403;

// --- 500 - 599: Optimizers ---
const int OP_SGD_UPDATE               = 500;
const int OP_ADAM_UPDATE              = 502;
const int OP_CLIP_GRAD_VALUE          = 504;

// --- 600 - 699: Reductions ---
const int OP_SUM_REDUCE               = 600;
const int OP_SUM_REDUCE_BACKWARD      = 601;
const int OP_SUM_REDUCE_COLUMNS       = 602;
const int OP_SUM_REDUCE_ROWS          = 604;
const int OP_EMBEDDING_FORWARD        = 606;
const int OP_EMBEDDING_BACKWARD       = 607;

// --- 700 - 799: Tensor Manipulation ---
const int OP_SLICE_ROW                = 700;
const int OP_SLICE_ROW_BACKWARD       = 701;
const int OP_SLICE_COLUMN             = 702;
const int OP_SLICE_COLUMN_BACKWARD    = 703;
const int OP_STACK_ROWS               = 704;
const int OP_STACK_ROWS_BACKWARD      = 705;
const int OP_CONCATENATE              = 706;
const int OP_CONCATENATE_BACKWARD     = 707;
const int OP_PAD2D                    = 708;
const int OP_PAD2D_BACKWARD           = 709;

// --- 800 - 999: Advanced Spatial & Sequence Layers ---
const int OP_CONV2D_FORWARD               = 800;
const int OP_CONV2D_BACKWARD_INPUT        = 801;
const int OP_CONV2D_BACKWARD_KERNEL       = 803;
const int OP_CONV2D_MULTI_FORWARD         = 804;
const int OP_CONV2D_MULTI_BACKWARD_INPUT  = 805;
const int OP_CONV2D_MULTI_BACKWARD_WEIGHT = 807;
const int OP_IM2COL                       = 808;
const int OP_COL2IM                       = 809;
const int OP_MAX_POOL_1D_FORWARD          = 810;
const int OP_MAX_POOL_1D_BACKWARD         = 811;
const int OP_MAX_POOL_2D_FORWARD          = 812;
const int OP_MAX_POOL_2D_BACKWARD         = 813;
const int OP_AVG_POOL_2D_FORWARD          = 814;
const int OP_AVG_POOL_2D_BACKWARD         = 815;
const int OP_GLOBAL_AVG_POOL_FORWARD      = 816;
const int OP_GLOBAL_AVG_POOL_BACKWARD     = 817;
const int OP_BATCH_NORM_1D_FORWARD        = 820;
const int OP_BATCH_NORM_1D_BACKWARD       = 821;
const int OP_BATCH_NORM_2D_FORWARD        = 822;
const int OP_BATCH_NORM_2D_BACKWARD       = 823;
const int OP_LAYER_NORM_FORWARD           = 824;
const int OP_LAYER_NORM_BACKWARD          = 825;
const int OP_DROPOUT_FORWARD              = 826;
const int OP_DROPOUT_BACKWARD             = 827;

const int OP_MARKOV_COUNT             = 900;
const int OP_MARKOV_NORMALIZE         = 902;
const int OP_MARKOV_PREDICT           = 904;

// --- 1000+: Fused Kernels ---
const int OP_MATMUL_BIAS_RELU_FORWARD = 1200;

// Transformer Specific codes
const int OP_RMS_NORM_FORWARD = 2000;
const int OP_RMS_NORM_BACKWARD = 2001;
const int OP_CAUSAL_MASK_FORWARD = 2002;
const int OP_CAUSAL_MASK_BACKWARD = 2003;
const int OP_ROPE_FORWARD = 2004;
const int OP_ROPE_BACKWARD = 2005;
const int OP_CROSS_ENTROPY_FORWARD = 2006;
const int OP_CROSS_ENTROPY_BACKWARD = 2007;
const int OP_ARGMAX_FORWARD = 2008;
