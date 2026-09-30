
![flutter_ml](https://raw.githubusercontent.com/NoirsCodingCorner/flutter_ml/master/doc/flutterML.png)

A deep learning library for Dart and Flutter. It brings native hardware-accelerated machine learning directly to your device. Train models and run inference locally without Python, cloud APIs, or static binaries.

## 🟥 Why flutter_ml?

Most machine learning in Flutter relies on external Python servers or read-only TFLite models. This package provides a native alternative. You get full control over tensor math, autograd graphs, and custom model architectures.

* **Zero Frame Drops:** Standard Dart lists trigger huge garbage collection pauses. This engine pre-allocates memory in native C heap or GPU VRAM. Static tape unrolling reuses buffers. Zero allocations happen during the training loop.
* **Compiled Execution Tapes:** FFI calls are expensive. Calling native code for every math operation introduces massive overhead. This library records operations into a byte-encoded tape. The entire graph runs on the GPU in a single dispatch.
* **On-Device Training:** Keep user data private. Train directly on the device.
* **Hugging Face Compatibility:** Load and export standard `.safetensors` model weights natively. No Python conversion scripts are needed.
* **Modern Architectures:** Build Transformers, Multi-Head Attention, RoPE, and spatio-temporal ConvLSTMs entirely in Dart.

## 🟧 CPU Engine (Eager Execution)

The CPU engine uses an eager computation graph. It is highly transparent. Use it for debugging, testing, or running lightweight models.

🟠 *Does not require specified Device architecture*
```dart
import 'package:flutter_ml/full_library.dart';

void main() {
  List<Layer<dynamic, dynamic>> layers = <Layer<dynamic, dynamic>>[];
  layers.add(DenseLayer(8, activation: ReLU()));
  layers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork model = SNetwork(layers, name: 'XOR-Model');

  List<Vector> inputs = <Vector>[];
  inputs.add(<double>[0.0, 0.0]);
  inputs.add(<double>[0.0, 1.0]);
  inputs.add(<double>[1.0, 0.0]);
  inputs.add(<double>[1.0, 1.0]);

  List<Vector> targets = <Vector>[];
  targets.add(<double>[0.0]);
  targets.add(<double>[1.0]);
  targets.add(<double>[1.0]);
  targets.add(<double>[0.0]);

  SGD optimizer = SGD(model.parameters, learningRate: 0.1);
  model.compile(configuredOptimizer: optimizer);

  model.fit(inputs, targets, epochs: 1000, debug: true);
}


```

## 🟨 GPU Engine (Static Tape Compilation)

The GPU engine builds static execution tapes for extreme performance across multiple operating systems. It requires initialization. It uses `TapeLayer` structures to orchestrate VRAM.

The engine actively supports and bundles acceleration targets for:

🟡**Windows** (`Target.cuda`) via `.dll` binaries.


🟡**Linux** (`Target.cuda`) via `.so` binaries.


🟡**Android** (`Target.androidArm64`, `Target.androidX8664`) via bundled NDK objects.



```dart
import 'package:flutter_ml/full_library.dart';

void main() async {
  await GPUEngine.initialize(debug: false, target: Target.cuda); // Select targeted architecture

  List<TapeLayer> layers = <TapeLayer>[];
  layers.add(DenseReluTL(8));
  layers.add(DenseTL(1));

  GPUTensor<Matrix> input = GPUTensor<Matrix>.empty(<int>[4, 2]);
  GPUTensor<Matrix> target = GPUTensor<Matrix>.empty(<int>[4, 1]);

  SeqModel<Matrix, Matrix> model = SeqModel<Matrix, Matrix>(
    layers,
    input,
    target: target,
    lossFunction: mseMatrixGPU,
    optimizerBuilder: (List<GPUTensor> params) {
      return AdamGPU(params, 0.01);
    }
  );

  model.compile();

  List<double> rawX = <double>[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
  List<double> rawY = <double>[0.0, 1.0, 1.0, 0.0];

  for (int epoch = 0; epoch < 1000; epoch = epoch + 1) {
    model.runTraining(inputData: rawX, targetData: rawY);
  }

  model.runForward(inputData: rawX);
  model.inferResult!.toCpu();
  print(model.inferResult!.value);

  model.free();
  GPUEngine.dispose();
}


```

## Core Design Decisions

Version `3.0.0` introduces complete framework restructuring. Architecture is focused on memory safety, static compilation, and zero-allocation runtime.

* **Direct Pointer Memory:** Tensors allocate via native C `calloc` (CPU) or device VRAM (GPU). Bypasses Dart Garbage Collection entirely.
* **Dual Execution Modes:** Eager graph evaluation for CPU. Deferred static bytecode compilation for GPU.
* **Static Tape Unrolling:** Intermediate tensors pre-allocated during `build()`. Reused across iterations. Zero memory allocations during training loops.
* **Hybrid Interop:** GPU subgraphs encapsulated within CPU autograd nodes. Native pointer accumulation via `GPUEngine.addPointers`.
* **Safetensors I/O:** Native binary decoding/encoding of Hugging Face `.safetensors`. Zero external dependencies.

---

## 🟩 CPU Components

### CPU Layers (20)

AveragePoolingLayer, BatchNormalizationLayer, Conv2D, ConvLSTMLayer, DenseLayer, DropoutLayer, DualLSTM, EmbeddingLayer, FlattenLayer, GlobalAveragePoolingLayer, LSTMLayer, MaxPoolingLayer, MultiHeadAttentionLayer, MultiLSTMLayer, NormalizationLayer, PositionalEncodingLayer, ReLULayer, RNNLayer, SingleHeadAttentionLayer, TransformerEncodingLayer.

### CPU Math Operations (58)

add, add3D, addMatrix, addMatrixAndVector, addScalar, addScalarToMatrix, addVector, avgPool1d, avgPool2d, batchNorm1dMath, batchNorm2dMath, binaryCrossEntropy, concatenate, concatenate3D, concatenateMatricesByColumn, conv2d, dot, dropoutMatrixMath, dropoutVectorMath, elementWiseMultiply, elementWiseMultiply3D, elementWiseMultiplyMatrix, eluMatrix, eluVector, globalAveragePooling, leakyReluMatrix, leakyReluVector, matMul, matVecMul, maxPool1d, maxPool2d, mishMatrix, mishVector, mse, mseMatrix, multiply, padMatrix, relu, reluMatrix, reshapeVectorToMatrix, scaleMatrix, selectRow, sigmoid, sigmoidMatrix, sigmoidScalar, softmaxMatrix, softmaxVector, softplus, stackMatricesTo3D, sum, sumMatrix, swishMatrix, swishVector, tanhMatrix, transpose, vectorExp, vectorLog, vectorTanh.

### CPU Activations (8)

ELU, LeakyReLU, Mish, ReLU, Sigmoid, SiLU, Softmax, Tanh.

### CPU Optimizers (8)

Adagrad, Adam, AdamW, AMSGrad, NAG, RMSprop, SGD, SGDMomentum.

---

## 🟦 GPU - Accelerated Components

Version `3.0.0` utilizes `CommandBuffer` execution tapes. Operations map to 32-bit `OpCodes`. Compiled graphs dispatch to CUDA 12.1+ devices in a single FFI call.

For full architectural blueprints, see the [GPU Tensor Documentation](https://github.com/NoirsCodingCorner/flutter_ml/blob/master/doc/gpu_tensor.md) and [Tape Layer](https://github.com/NoirsCodingCorner/flutter_ml/blob/master/doc/tapeLayer.md).

### GPU TapeLayers (29)

AveragePooling2DGPU, BatchNorm1DGPU, BatchNorm2DGPU, Conv2DTapeLayer, ConvLSTMTapeLayer, DenseTL, DenseReluTL, DropoutMatrixTapeLayer, DropoutTapeLayer, DualLSTMTapeLayer, EmbeddingMatrixTapeLayer, EmbeddingTapeLayer, FlattenTapeLayer, GeluLayerMatrixTapeLayer, GeluLayerTapeLayer, GlobalAveragePooling1DTapeLayer, GlobalAveragePoolingGPU, LayerNormalizationTapeLayer, LSTMTapeLayer, MaxPooling1DTapeLayer, MaxPooling2DTapeLayer, MultiHeadAttentionTapeLayer, PositionalEncodingTapeLayer, ReLULayerMatrixTapeLayer, ReLULayerTapeLayer, RNNTapeLayer, SigmoidMatrixTapeLayer, SingleHeadAttentionTapeLayer, TransformerEncoderBlockTapeLayer.

### GPU Math Operations (81)

absGPU, add3DGPU, addBiasToFeatureMapGPU, addBiasToMatMulOutGPU, addGPU, addMatrixAndVectorGPU, addMatrixGPU, addScalar3DGPU, addScalarMatrixGPU, addScalarVectorGPU, addVectorGPU, applyRopeGPU, argmaxGPU, avgPool2dGPU, batchNorm1dGPU, batchNorm2dGPU, binaryCrossEntropyGPU, broadcastAddVectorToMatrixGPU, buildMarkovTableGPU, causalMaskGPU, clampGPU, concatenate3DGPU, concatenateGPU, concatenateMatricesByColumnGPU, conv2dMultiChannelGPU, conv2dSimpleGPU, cosineSimilarityGPU, crossEntropyLossGPU, divide3DGPU, divideGPU, divideMatrixGPU, divideVectorGPU, dotProductGPU, dropoutGPU, elementWiseMultiply3DGPU, elementWiseMultiplyGPU, elementWiseMultiplyMatrixGPU, embeddingLookupBatchGPU, embeddingLookupGPU, euclideanDistanceGPU, flatten3DToMatrixGPU, geluGPU, geluMatrixGPU, globalAveragePoolingGPU, im2colGPU, l2NormGPU, layerNormMatrixGPU, loadSampleGPU, logGPU, maeLossGPU, markovPredictGPU, matMulBiasReluGPU, matMulGPU, matVecMulGPU, maxPool1dGPU, maxPool2dGPU, mseGPU, mseMatrixGPU, multiplyGPU, multiplyScalarGPU, padMatrixGPU, powGPU, reluGPU, reluMatrixGPU, reshape3DToMatrixGPU, reshapeMatrixTo3DGPU, reshapeVectorToMatrixGPU, rmsNormMatrixGPU, scaleMatrixGPU, scatterHeadsGPU, selectMatrixFrom3DGPU, selectRowGPU, sigmoid3DGPU, sigmoidGPU, sigmoidMatrixGPU, sigmoidScalarGPU, sliceColumnGPU, softmaxMatrixGPU, sqrtGPU, stackMatricesGPU, subtract3DGPU, subtractGPU, subtractMatrixGPU, subtractVectorGPU, sumGPU, sumMatrixGPU, sumReduceColumnsGPU, sumReduceRowsGPU, tanh3DGPU, tanhMatrixGPU, transposeGPU, vectorExpGPU, vectorTanhGPU.

### GPU Optimizers (2)

AdamGPU, SGDGPU.

---

## Static Graph Compilation Example

The `SeqModel` object replaces eager execution loops. It handles VRAM lifecycle and compiles reusable execution tapes (`trainForward`, `trainBackward`, `trainOptimize`, `inferTape`).

```dart
import 'package:flutter_ml/full_library.dart';

void main() async {
  // Initialize Native Engine
  await GPUEngine.initialize(debug: false, target: Target.cuda);

  // Data mapping
  List<double> rawX = <double>[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
  List<double> rawY = <double>[0.0, 1.0, 1.0, 0.0];

  // Define Architecture
  List<TapeLayer> layers = <TapeLayer>[];
  layers.add(DenseReluTL(8));
  layers.add(DenseTL(1));

  // Define I/O Boundaries
  GPUTensor<Matrix> input = GPUTensor<Matrix>.empty(<int>[4, 2]);
  GPUTensor<Matrix> target = GPUTensor<Matrix>.empty(<int>[4, 1]);

  // Build Sequential Compiler
  SeqModel<Matrix, Matrix> net = SeqModel<Matrix, Matrix>(
    layers,
    input,
    target: target,
    lossFunction: mseMatrixGPU,
    optimizerBuilder: (List<GPUTensor> params) => AdamGPU(params, 0.01)
  );

  // Allocate VRAM and Compile Tapes
  net.compile();
  
  // High-Speed Training Loop
  for (int epoch = 1; epoch <= 10000; epoch = epoch + 1) {
    net.runTraining(inputData: rawX, targetData: rawY);
  }

  // Compile Inference Tape & Predict
  net.predict(input);
  net.runForward(inputData: rawX);
  
  // Synchronize VRAM to CPU
  net.inferResult!.toCpu();
  print(net.inferResult!.value);

  // Cleanup
  net.free();
  GPUEngine.dispose();
}


```

---

## 🟪 Benchmark Speed on Consumer Hardware

**Test Environment**

* **Hardware:** RTX 3060 12GB
* **Backend:** CUDA 12.1 (Tensor Core acceleration enabled)
* **Workload Size:** Vector operations evaluated on 134,217,728 elements (~537 MB); Matrix operations evaluated on 8192x8192 dimensions.
* **Methodology:** 50 iterations per operation. VRAM is aggressively wiped and reallocated between each run to simulate cold loading and tape compilation times.

**Metric Descriptions**

* **Operation:** The executed mathematical or layer function.
* **Time (ms):** Pure native execution time on the GPU after the tape is compiled.
* **Bandwidth (GB/s):** Memory throughput achieved during the operation.
* **Compute (TFLOPs):** Theoretical processing overhead and throughput measured in TeraFLOPS.
* **Alloc/Tape (ms):** Time required for the CPU to allocate VRAM and compile the execution tape. (Note: In standard training/inference, this is a one-time cost).
* **Free (ms):** Time required to release VRAM and destroy native pointers.

| Operation | Time (ms) | Bandwidth (GB/s) | Compute (TFLOPs) | Alloc/Tape (ms) | Free (ms) |
| --- | --- | --- | --- | --- | --- |
| **ADD** | 5.24 | 307.31 | 0.0256 | 2135.94 | 38.08 |
| **MARKOV_TBL** | 11.98 | 22.40 | 0.0056 | 648.10 | 16.40 |
| **MARKOV_PRD** | 3.38 | 213.20 | 0.0474 | 1271.30 | 26.51 |
| **SUBTRACT** | 5.23 | 308.12 | 0.0257 | 2256.46 | 42.87 |
| **MULTIPLY** | 5.14 | 313.05 | 0.0261 | 2466.38 | 15.88 |
| **DIVIDE** | 5.39 | 298.92 | 0.0249 | 2129.76 | 48.59 |
| **ABS** | 3.60 | 298.63 | 0.0373 | 1704.05 | 33.45 |
| **SQRT** | 3.61 | 297.78 | 0.0372 | 1794.70 | 35.15 |
| **LOG** | 4.71 | 227.73 | 0.0285 | 2335.28 | 33.97 |
| **POW** | 3.63 | 295.84 | 0.0370 | 1488.77 | 30.05 |
| **CLAMP** | 3.66 | 293.12 | 0.0366 | 1673.10 | 31.99 |
| **MATMUL** | 81.23 | 9.91 | 13.5354 | 1373.70 | 17.80 |
| **TRANSPOSE** | 2.38 | 225.77 | 0.0000 | 1059.03 | 24.71 |
| **MAT_VEC** | 0.87 | 307.53 | 0.1537 | 379.82 | 5.80 |
| **ADD_BIAS** | 1.93 | 277.69 | 0.0347 | 710.38 | 12.69 |
| **SCALE_MAT** | 1.99 | 270.11 | 0.0338 | 1107.48 | 17.39 |
| **ADD_SCALAR** | 3.53 | 303.96 | 0.0380 | 1708.55 | 19.71 |
| **RELU** | 4.19 | 256.38 | 0.0320 | 2446.01 | 28.16 |
| **SIGMOID** | 4.87 | 220.38 | 0.0826 | 2385.56 | 12.14 |
| **TANH** | 4.46 | 240.48 | 0.0902 | 2415.69 | 11.53 |
| **GELU** | 5.33 | 201.43 | 0.1259 | 3790.56 | 32.06 |
| **SOFTMAX** | 3.12 | 258.28 | 0.0646 | 1023.32 | 8.64 |
| **BCE_LOSS** | 24.06 | 44.64 | 0.0223 | 1405.21 | 12.10 |
| **MSE_VEC** | 24.40 | 44.01 | 0.0165 | 1410.52 | 10.48 |
| **MSE_MAT** | 12.11 | 44.33 | 0.0166 | 711.45 | 5.07 |
| **SUM_VEC** | 1.82 | 294.55 | 0.0736 | 701.77 | 17.40 |
| **SUM_COLS** | 0.89 | 300.85 | 0.0752 | 523.13 | 7.02 |
| **SUM_ROWS** | 7.13 | 37.65 | 0.0094 | 349.99 | 16.54 |
| **EMBED_VEC** | 15.30 | 421.33 | 0.0000 | 5286.45 | 84.81 |
| **EMBED_MAT** | 15.26 | 422.36 | 0.0000 | 4544.87 | 88.81 |
| **SLICE_COL** | 1.00 | 269.41 | 0.0000 | 962.99 | 14.19 |
| **SLICE_ROW** | 0.11 | 76.73 | 0.0000 | 1174.23 | 5.80 |
| **SLICE_3D** | 0.08 | 99.27 | 0.0000 | 370.54 | 3.04 |
| **CONCAT_VEC** | 5.39 | 199.13 | 0.0000 | 2857.50 | 30.98 |
| **STACK_MAT** | 0.57 | 235.83 | 0.0000 | 224.28 | 3.67 |
| **SCAT_HEADS** | 0.33 | 75.67 | 0.0000 | 36.79 | 1.72 |
| **PAD_2D** | 0.47 | 278.27 | 0.0000 | 194.99 | 0.97 |
| **LAYER_NORM** | 3.47 | 154.71 | 0.1547 | 727.34 | 5.75 |
| **DROPOUT** | 2.75 | 293.00 | 0.0244 | 1073.25 | 18.37 |
| **DOT_PROD** | 7.31 | 293.90 | 0.0367 | 2356.80 | 32.84 |
| **L2_NORM** | 5.66 | 284.70 | 0.0475 | 1797.15 | 14.86 |
| **EUC_DIST** | 11.90 | 270.64 | 0.0338 | 3751.22 | 34.63 |
| **COS_SIM** | 18.41 | 291.55 | 0.0437 | 3780.90 | 31.48 |
| **MAE_LOSS** | 48.99 | 65.75 | 0.0082 | 3310.34 | 29.07 |

## Future Plans

1. Integration of complete pre-trained LLM pipelines.
2. Additional science and custom kernel mapping.

