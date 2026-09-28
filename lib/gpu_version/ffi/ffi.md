# FFI to GPU Accelerated Backend

Communication between Dart and the high-performance mathematical execution engine is handled via a Foreign Function Interface (FFI). 
This bridge relies on core components to encode instructions and manage memory efficiently. 
The FFI module is structured around four primary pillars: `GPUEngine`, `CommandBuffer`, `TapeDecoder`, and `OpCodes`.

## 1. GPUEngine (The Native Executor)

The `GPUEngine` manages communication with native runtimes (C++, CUDA, WebGPU) and acts as the bridge between Dart and the FFI bindings. 
It handles the lifecycle of the execution engine, manages memory transfers, and triggers command execution.

**Initialization & Lifecycle**
* **`initialize({bool debug = false, Target target = Target.cuda})`**: 
Loads the native dynamic library based on the provided `Target` (`android_arm64`, `android_x86_64`, or `cuda`), binds the native functions, and sets the static `handle` for the session.
* **`dispose()`**: Frees the native executor instance associated with the current `handle` and resets the handle to `0`.



**Memory Management (Host to Device)**

* **`load(String name, Pointer<Float> data, List<int> shape)`**: Allocates and transfers tensor data from the host CPU to the device GPU.


* **`retrieve(String name, Pointer<Float> destination)`**: Transfers tensor data from the device GPU back to the host, populating the destination pointer.


* **`initRandomUniform(String name, double scale, int seed)`**: Initializes an allocated tensor on the device with uniformly distributed random numbers.


* **`free(String name)`**: Deallocates a specific tensor's memory natively.


* **`getTensorNames()`**: Returns a `List<String>` of all tensor names currently allocated on the GPU.



**Execution Dispatch & Utilities**

* **`run(Uint8List tapeBytes)`**: Dispatches a compiled command tape to the native engine for execution.


* **`addPointers(Pointer<Float> dest, Pointer<Float> src, int length)`**: Allows a direct memory copy from a source to a destination pointer for fast transfers.



## 2. CommandBuffer (The Execution Tape)

The `CommandBuffer` records mathematical operations as a continuous stream of raw bytes, acting as a tape for deferred backend execution.

**Command Anatomy & Memory Layout**
Commands contain zero padding to minimize FFI overhead and follow a strict two-part linear layout:

1. **OpCode (4 bytes)**: A 32-bit integer designating the specific operation to execute.


2. **Arguments (Variable Length)**: Inputs required by the operation.


* **Int / Float**: 4 bytes (32-bit).


* **Bool**: 1 byte (8-bit).


* **String**: 2-byte length prefix (16-bit unsigned integer) followed directly by UTF-8 bytes.





Example memory block for a command with three string arguments:

```text
[OpCode (4 bytes)] [Len A (2 bytes)] [Text A] [Len B (2 bytes)] [Text B] [Len Out (2 bytes)] [Text Out]

```

## 3. TapeDecoder (Debugging & Visualization)

The `TapeDecoder` is a debugging utility designed to translate the raw byte contents of a `CommandBuffer` into a human-readable visual execution graph.

**Usage**
To inspect a buffer, extract its bytes and trigger the decoder:

```dart
TapeDecoder decoder = TapeDecoder(commandBuffer.bytes());
decoder.decode();

```

This reads the OpCodes sequentially and outputs a formatted log via the `Logger` class, detailing exactly which operations were dispatched and in what order.

## 4. OpCodes (The Instruction Set)

OpCodes are 32-bit integers that map directly to C++ engine instructions. They dictate which native mathematical function is executed when the tape is processed.

**Categorization Overview**
To view the specific integer mappings, reference the `OpCodes.dart` file. The codes are logically grouped by functionality:

* **0 - 99**: Data & Memory Management (Load, Store, Copy).


* **100 - 199**: Basic Math (Add, Sub, Element-wise functions like Exp, Log, Sqrt).


* **200 - 299**: Matrix Operations (Matmul, Transpose, Scale).


* **300 - 399**: Activations (ReLU, Sigmoid, Tanh, GELU, Softmax).


* **400 - 499**: Loss Functions (MSE, BCE).


* **500 - 599**: Optimizers (SGD, Adam).


* **600 - 699**: Reductions (Sum Reduce, Embedding).


* **700 - 799**: Tensor Manipulation (Slice, Stack, Concatenate, Pad2D).


* **800 - 999**: Advanced Spatial & Sequence Layers (Conv2D, MaxPool, AvgPool, BatchNorm, Dropout).


* **1000+**: Fused Kernels (Matmul + Bias + ReLU).


* **2000+**: Transformer Specific operations (RMS Norm, Causal Mask, RoPE, Cross Entropy, Argmax).