# FFI to GPU Accelerated Backend

Communication between Dart and the high-performance mathematical execution engine is handled via a Foreign Function Interface (FFI). This bridge relies on several core components to encode instructions and manage memory efficiently.

---

## CommandBuffer

The `CommandBuffer` records and serializes mathematical operations so they can be interpreted by the C++ backend.

Commands in the buffer are encoded as a continuous, tightly packed stream of raw bytes using host endianness. To reduce the overhead of sending data via FFI while maintaining readability, there is zero padding and no object overhead.

### Command Anatomy
Every command strictly follows a two-part linear layout:

1. **OpCode (4 Bytes)**
   A 32-bit integer that tells the C++ engine which operation to execute (e.g., `100` for `OP_ADD`).
2. **Arguments (Variable Length)**
   The inputs required for the operation, written immediately after the OpCode in the exact order the engine expects:
    * **Int / Float:** 4 bytes (32-bit)
    * **Bool:** 1 byte (8-bit, `1` or `0`)
    * **String:** A 2-byte length prefix (16-bit unsigned integer) followed immediately by the raw UTF-8 text bytes.

### Memory Layout Example
If you dispatch an `add` command with three string arguments, the memory block is packed exactly like this:

```text
[OpCode (4 bytes)] [Len A (2 bytes)] [Text A] [Len B (2 bytes)] [Text B] [Len Out (2 bytes)] [Text Out]
```

---

## CudaEngine

The `CudaEngine` is the bridge to backend computation. It acts as a Dart FFI wrapper for the native C++ CUDA execution library (e.g., `cuda_executor.dll` on Windows or `libcuda_executor.so` on Linux). It manages the lifecycle of the execution engine, handles memory transfers between the host (CPU) and device (GPU), and triggers command execution.

### Lifecycle Management

*   `initialize({bool debug = false})`
    Loads the native dynamic library, binds all necessary C/C++ functions to Dart, and creates a new native executor instance, saving its ID to the static `handle`.
*   `dispose()`
    Frees the entire native executor instance associated with the current `handle` and resets the handle to `0`.

### Memory & Tensor Management

*   `load(String name, Pointer<Float> data, List<int> shape)`
    Allocates and transfers tensor data from Host to Device (H2D). It takes the tensor's name, a pointer to the flat float data, and its shape dimensions.
*   `retrieve(String name, Pointer<Float> destination)`
    Transfers tensor data from Device back to Host (D2H), populating the provided `destination` pointer with the values of the requested tensor.
*   `initRandomUniform(String name, double scale, int seed)`
    Directs the native engine to initialize an existing tensor with uniformly distributed random numbers, scaled by the `scale` factor.
*   `free(String name)`
    Deallocates the memory associated with a specific tensor by name on the native side.
*   `getTensorNames()`
    Returns a `List<String>` containing the names of all tensors currently held in the native engine's memory.

### Execution

*   `run(Uint8List tapeBytes)`
    Passes a compiled command buffer tape (an array of bytes containing OpCodes and arguments) to the native engine for execution. The bytes are copied into native memory before dispatch.

### Utilities

*   `addPointers(Pointer<Float> dest, Pointer<Float> src, int length)`
    A direct memory utility that exposes a native function to add the contents of a source pointer into a destination pointer over a specified length.</Float></Float></String></Float></Float>


# OpCodes

The primary way of interaction as mentioned are commands encoded into OPCodes (Operational Commands).
As of May 1st the command list of the Engine natively supports the following functions with their corresponding codes:

```dart
// =========================================================================
// C++ Engine Instruction Map (Byte Tape OpCodes)
// Ensure these perfectly match the C++ OpCode enum in manager.h
// if not, please check versions of the current dart and cuda engine
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
```




























