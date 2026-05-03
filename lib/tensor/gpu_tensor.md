## GPU `Tensor` Mathematics

The library's GPU-accelerated autograd engine is built upon two fundamental classes: `GPUTensor` and `GPUNode`. While conceptually similar to their pure Dart equivalents, these classes act as bridges to the native CUDA backend. The `GPUTensor` class manages handles to VRAM rather than holding large multidimensional arrays in Dart memory, and the `GPUNode` class represents operations that encode instructions onto a command buffer tape.

---

### The `GPUTensor` Class

The `GPUTensor` is the primary data structure, acting as a lightweight Dart container that tracks shapes and identifies VRAM allocations managed by the `CudaEngine`.

**Supported Data Types:**
A `GPUTensor<T>` determines its shape automatically based on the provided initial value, flattening it into a 1D VRAM buffer. It supports the extraction of:
*   `Scalar` (0D, initialized with `double`)
*   `Vector` (1D, initialized with `List<double>`)
*   `Matrix` (2D, initialized with `List<List<double>>`)
*   `Tensor3D` (3D, initialized with `List<List<List<double>>>`)

**Properties:**
A `GPUTensor` instance holds several key properties:
*   **`id`**: A unique string identifier (e.g., `t_gpu_0`) used to reference the specific tensor data and gradient inside the native engine's memory.
*   **`shape`**: A list of integers representing the dimensions of the tensor.
*   **`data` & `grad`**: Local 1D CPU buffers that remain empty until explicitly synchronized from the GPU.
*   **`creator`**: An optional `GPUNode` object referencing the operation that produced this tensor.

**Initialization & VRAM Allocation:**
When a `GPUTensor` is created, it automatically triggers a direct FFI call to allocate an empty block of VRAM for both its data and its gradient (using `<id>_grad`). It completely bypasses deep Dart list construction when initialized via `GPUTensor.empty()` or `GPUTensor.randomUniform()` to maximize performance.

**Synchronization Methods:**
Because the numerical data lives on the GPU, you cannot read it instantly.
*   **`toCpu()`**: Must be called before reading values. This method retrieves the flat `Float32List` arrays from the native engine and stores them in the local `data` and `grad` properties.
*   **`value` & `gradValue`**: Getters that internally call `_unflatten()` to reconstruct the flat CPU mirrors back into their typed, multidimensional Dart formats (like `List<List<double>>`). If `toCpu()` hasn't been called, these will throw an exception.
*   **`free()`**: Instructs the `CudaEngine` to deallocate the tensor and its gradient from GPU memory.

---

### The `GPUNode` Class

The `GPUNode` class represents a single operation within the computational graph, storing the necessary inputs and logic to generate backward-pass instructions.

**Properties:**
*   **`inputs`**: A `List<GPUTensor>` containing the tensors used as inputs.
*   **`backwardFn`**: A function taking a `CommandBuffer` as an argument. Instead of performing math, this closure packs byte-encoded OpCodes onto a tape to execute the chain rule later.
*   **`opName`**: A string identifier for debugging and graph visualization.
*   **`cost`**: An estimated computational cost.
*   **`extraParams`**: A map storing any static parameters required during the backward step.

---

### How It Works: The Forward Pass

During the forward pass, GPU mathematical functions (like `addGPU` or `matMulGPU`) do not execute math in Dart. Instead, they:
1.  **Allocate VRAM**: Create a new `GPUTensor.empty` based on the predicted output shape.
2.  **Record the Tape**: Write an execution OpCode (e.g., `OP_ADD`) to the active forward `CommandBuffer`, along with the `id` strings of the inputs and output.
3.  **Build Graph**: Assign a `GPUNode` to the output tensor's `creator`, defining exactly how to encode the derivative onto a future backward tape.

---

### How It Works: The Backward Pass

The backward pass does not execute instantly. It is a compilation step triggered by calling `backward(CommandBuffer tape)` on an output tensor.

```dart
CommandBuffer backTape = CommandBuffer();
lossTensor.backward(backTape);
CudaEngine.run(backTape.bytes());
```

The `backward()` method executes the following logic:
1.  **Gradient Seeding**: If `fillOnes` is true, it writes an `OP_FILL` command to the tape, instructing the GPU to fill the starting tensor's gradient memory with `1.0`.
2.  **Topological Sort**: It traverses the `creator` links backward to build a `topo` list of all `GPUNode` objects, ensuring the correct order of dependency.
3.  **Tape Compilation**: It iterates through the `topo` list in reverse. For each `GPUNode`, it executes the `backwardFn(backwardTape)` closure. This packs specific backward derivative OpCodes (e.g., `OP_ADD_INTO`, `OP_MATMUL_BACKWARD`) onto the buffer tape.

Finally, the compiled tape bytes must be passed to `CudaEngine.run()` to actually perform the gradient calculations natively on the GPU device.

The available mathematical operations are: 

### Data Management
*   `GPUTensor<Matrix> reshapeVectorToMatrixGPU(GPUTensor<Vector> v, int numRows, int numCols, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> reshapeMatrixTo3DGPU(GPUTensor<Matrix> m, int c, int h, int w, CommandBuffer tape)`
*   `GPUTensor<Matrix> reshape3DToMatrixGPU(GPUTensor<Tensor3D> t, int rows, int cols, CommandBuffer tape)`
*   `GPUTensor<Matrix> flatten3DToMatrixGPU(GPUTensor<Tensor3D> t, CommandBuffer tape)`
*   `GPUTensor<Vector> loadSampleGPU(GPUTensor<Matrix> dataset, int sampleIndex, CommandBuffer tape)`

### Basic Math
*   `GPUTensor<T> addGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape)`
*   `GPUTensor<Vector> addVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> addMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> add3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape)`
*   `GPUTensor<T> subtractGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape)`
*   `GPUTensor<Vector> subtractVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> subtractMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> subtract3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape)`
*   `GPUTensor<T> multiplyGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape)`
*   `GPUTensor<Scalar> multiplyScalarGPU(GPUTensor<Scalar> a, GPUTensor<Scalar> b, CommandBuffer tape)`
*   `GPUTensor<Vector> elementWiseMultiplyGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> elementWiseMultiply3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> elementWiseMultiplyMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<T> divideGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape)`
*   `GPUTensor<Vector> divideVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> divideMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> divide3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape)`
*   `GPUTensor<Vector> vectorExpGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<T> absGPU<T>(GPUTensor<T> a, CommandBuffer tape)`
*   `GPUTensor<T> sqrtGPU<T>(GPUTensor<T> a, CommandBuffer tape)`
*   `GPUTensor<T> logGPU<T>(GPUTensor<T> a, CommandBuffer tape)`
*   `GPUTensor<T> powGPU<T>(GPUTensor<T> a, double exponent, CommandBuffer tape)`
*   `GPUTensor<T> clampGPU<T>(GPUTensor<T> a, double minVal, double maxVal, CommandBuffer tape)`

### Matrix Operations
*   `GPUTensor<Matrix> matMulGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<Vector> matVecMulGPU(GPUTensor<Matrix> mMat, GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> transposeGPU(GPUTensor<Matrix> a, CommandBuffer tape)`
*   `GPUTensor<Matrix> addMatrixAndVectorGPU(GPUTensor<Matrix> m, GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> addScalarMatrixGPU(GPUTensor<Matrix> m, GPUTensor<Scalar> b, CommandBuffer tape)`
*   `GPUTensor<Vector> addScalarVectorGPU(GPUTensor<Vector> v, double scalar, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> addScalar3DGPU(GPUTensor<Tensor3D> t, double scalar, CommandBuffer tape)`
*   `GPUTensor<Matrix> addBiasToFeatureMapGPU(GPUTensor<Matrix> m, GPUTensor<Matrix> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> addBiasToMatMulOutGPU(GPUTensor<Matrix> m, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> broadcastAddVectorToMatrixGPU(GPUTensor<Matrix> m, GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> scaleMatrixGPU(GPUTensor<Matrix> m, double s, CommandBuffer tape)`

### Activations
*   `GPUTensor<Vector> reluGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> reluMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<T> sigmoidScalarGPU<T>(GPUTensor<T> s, CommandBuffer tape)`
*   `GPUTensor<Vector> sigmoidGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> sigmoidMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> sigmoid3DGPU(GPUTensor<Tensor3D> t, CommandBuffer tape)`
*   `GPUTensor<Vector> vectorTanhGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> tanhMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> tanh3DGPU(GPUTensor<Tensor3D> t, CommandBuffer tape)`
*   `GPUTensor<Vector> geluGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Matrix> geluMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<Matrix> softmaxMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`

### Loss Functions
*   `GPUTensor<Scalar> binaryCrossEntropyGPU<T>(GPUTensor<T> prediction, GPUTensor<T> target, CommandBuffer tape)`
*   `GPUTensor<Scalar> mseGPU(GPUTensor<Vector> predictions, GPUTensor<Vector> targets, CommandBuffer tape)`
*   `GPUTensor<Scalar> mseMatrixGPU(GPUTensor<Matrix> predictions, GPUTensor<Matrix> targets, CommandBuffer tape)`

### Optimizers
*   `void sgdUpdateGPU(GPUTensor<dynamic> data, double lr, CommandBuffer tape)`
*   `void adamUpdateGPU(GPUTensor<dynamic> data, GPUTensor<dynamic> m, GPUTensor<dynamic> v, double lr, double beta1, double beta2, double eps, int step, double weightDecay, CommandBuffer tape)`
*   `void clipGradValueGPU(GPUTensor<dynamic> tensor, double clipValue, CommandBuffer tape)`

### Reductions
*   `GPUTensor<Scalar> sumGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Scalar> sumMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<Matrix> embeddingLookupGPU(GPUTensor<Vector> indices, GPUTensor<Matrix> weights, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> embeddingLookupBatchGPU(GPUTensor<Matrix> batchIndices, GPUTensor<Matrix> weights, CommandBuffer tape)`
*   `GPUTensor<Vector> sumReduceColumnsGPU(GPUTensor<Matrix> m, CommandBuffer tape)`
*   `GPUTensor<Vector> sumReduceRowsGPU(GPUTensor<Matrix> m, CommandBuffer tape)`

### Tensor Manipulations
*   `GPUTensor<Matrix> sliceColumnGPU(GPUTensor<Matrix> input, int startCol, int endCol, CommandBuffer tape)`
*   `GPUTensor<Vector> selectRowGPU(GPUTensor<Matrix> m, int rowIndex, CommandBuffer tape)`
*   `GPUTensor<Matrix> selectMatrixFrom3DGPU(GPUTensor<Tensor3D> t, int index, CommandBuffer tape)`
*   `GPUTensor<Vector> concatenateGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Matrix> concatenateMatricesByColumnGPU(List<GPUTensor<Matrix>> matrices, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> concatenate3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> stackMatricesGPU(List<GPUTensor<Matrix>> matrices, CommandBuffer tape)`
*   `GPUTensor<Matrix> scatterHeadsGPU(List<GPUTensor<Matrix>> heads, int dModel, CommandBuffer tape)`
*   `GPUTensor<Matrix> padMatrixGPU(GPUTensor<Matrix> input, int padSize, CommandBuffer tape)`

### Advanced Layers
*   `GPUTensor<Tensor3D> conv2dMultiChannelGPU(GPUTensor<dynamic> input, GPUTensor<Tensor3D> weight, GPUTensor<Vector> bias, int kH, int kW, CommandBuffer tape, {String padding = 'valid', int strideH = 1, int strideW = 1})`
*   `GPUTensor<Matrix> conv2dSimpleGPU(GPUTensor<Matrix> input, GPUTensor<Matrix> kernel, CommandBuffer tape)`
*   `GPUTensor<Matrix> im2colGPU(GPUTensor<dynamic> input, int kH, int kW, CommandBuffer tape)`
*   `GPUTensor<Vector> maxPool1dGPU(GPUTensor<Vector> input, int poolSize, int stride, CommandBuffer tape)`
*   `GPUTensor<Matrix> maxPool2dGPU(GPUTensor<Matrix> input, int poolSize, int stride, CommandBuffer tape)`
*   `GPUTensor<Matrix> avgPool2dGPU(GPUTensor<Matrix> input, int poolSize, int stride, CommandBuffer tape)`
*   `GPUTensor<Vector> globalAveragePoolingGPU(GPUTensor<Matrix> input, CommandBuffer tape)`
*   `GPUTensor<Vector> batchNorm1dGPU(GPUTensor<Vector> input, GPUTensor<Vector> gamma, GPUTensor<Vector> beta, GPUTensor<Vector> runningMean, GPUTensor<Vector> runningVariance, double momentum, double epsilon, bool isTraining, CommandBuffer tape)`
*   `GPUTensor<Tensor3D> batchNorm2dGPU(GPUTensor<Tensor3D> input, GPUTensor<Vector> gamma, GPUTensor<Vector> beta, GPUTensor<Vector> runningMean, GPUTensor<Vector> runningVariance, double momentum, double epsilon, bool isTraining, CommandBuffer tape)`
*   `GPUTensor<Matrix> layerNormMatrixGPU(GPUTensor<Matrix> m, GPUTensor<Vector> gamma, GPUTensor<Vector> beta, GPUTensor<Vector> meanCache, GPUTensor<Vector> rstdCache, double epsilon, CommandBuffer tape)`
*   `GPUTensor<T> dropoutGPU<T>(GPUTensor<T> input, double rate, CommandBuffer tape)`
*   `GPUTensor<Matrix> buildMarkovTableGPU(GPUTensor<Vector> sequence, int order, int numStates, CommandBuffer tape)`
*   `GPUTensor<Matrix> markovPredictGPU(GPUTensor<Matrix> historyBatch, GPUTensor<Matrix> probTable, int numStates, CommandBuffer tape)`

### Fused Kernels
*   `GPUTensor<Matrix> matMulBiasReluGPU(GPUTensor<Matrix> x, GPUTensor<Matrix> w, GPUTensor<Vector> b, CommandBuffer tape, List<GPUTensor> intermediates)`
*   `GPUTensor<Scalar> dotProductGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Scalar> l2NormGPU(GPUTensor<Vector> v, CommandBuffer tape)`
*   `GPUTensor<Scalar> euclideanDistanceGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Scalar> cosineSimilarityGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape)`
*   `GPUTensor<Scalar> maeLossGPU(GPUTensor<Vector> preds, GPUTensor<Vector> targets, CommandBuffer tape)`