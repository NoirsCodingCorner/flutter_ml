import 'dart:typed_data';
import 'package:flutter_ml/full_library.dart'; // Adjust based on your actual imports

class SNetworkGPU {
  List<TapeLayer> layers = <TapeLayer>[];
  List<GPUTensor> allParams = <GPUTensor>[];
  List<GPUTensor> intermediates = <GPUTensor>[];

  late GPUTensor<Matrix> inputRef;
  late GPUTensor<Matrix> targetRef;
  late GPUTensor<dynamic> outputRef;
  late GPUTensor<Scalar> lossRef;

  late OptimizerGPU optimizer;

  late Uint8List fTape;
  late Uint8List bTape;
  late Uint8List zTape;
  late Uint8List oTape;

  SNetworkGPU();


  void add(TapeLayer layer) {
    layers.add(layer);
  }

  void compile(List<int> inputShape, List<int> targetShape, double learningRate) {
    // 1. Create dummy input and target references to establish VRAM footprint
    inputRef = GPUTensor<Matrix>.empty(inputShape);
    targetRef = GPUTensor<Matrix>.empty(targetShape);

    // 2. Build Layers & Cascade Shapes
    GPUTensor<dynamic> current = inputRef;
    CommandBuffer buildTape = CommandBuffer(); // Throwaway tape
    List<GPUTensor> buildTrash = <GPUTensor>[]; // Throwaway intermediates

    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].build(current);
      current = layers[i].forward(current, buildTape, buildTrash);
    }

    for (int i = 0; i < buildTrash.length; i = i + 1) {
      buildTrash[i].free();
    }

    // 3. Populate allParams
    allParams.clear();
    for (int i = 0; i < layers.length; i = i + 1) {
      List<GPUTensor> params = layers[i].parameters;
      for (int j = 0; j < params.length; j = j + 1) {
        allParams.add(params[j]);
      }
    }

    // 4. Instantiate the optimizer
    optimizer = SGDGPU(allParams, learningRate);

    // 5a. Trace Forward Tape (fTape)
    CommandBuffer fCommand = CommandBuffer();
    intermediates.clear();
    current = inputRef;

    for (int i = 0; i < layers.length; i = i + 1) {
      current = layers[i].forward(current, fCommand, intermediates);
    }

    outputRef = current;
    lossRef = mseMatrixGPU(outputRef as GPUTensor<Matrix>, targetRef, fCommand);
    fTape = fCommand.bytes();

    // 5b. Trace Backward & ZeroGrad Tape (bTape)
    CommandBuffer bCommand = CommandBuffer();

    optimizer.zeroGrad(bCommand);
    for (int i = 0; i < intermediates.length; i = i + 1) {
      bCommand.putInt(OP_ZERO_GRAD);
      bCommand.putString(intermediates[i].id + '_grad');
    }

    bCommand.putInt(OP_ZERO_GRAD);
    bCommand.putString(inputRef.id + '_grad');
    bCommand.putInt(OP_ZERO_GRAD);
    bCommand.putString(outputRef.id + '_grad');

    // Triggers backpropagation and automatically fills loss grad with 1.0
    lossRef.backward(bCommand, fillOnes: true);

    for (int i = 0; i < allParams.length; i = i + 1) {
      bCommand.putInt(OP_CLIP_GRAD_VALUE);
      bCommand.putString(allParams[i].id + '_grad');
      bCommand.putFloat(1.0);
    }
    bTape = bCommand.bytes();

    // 5c. Trace Optimize Tape (oTape)
    CommandBuffer oCommand = CommandBuffer();
    optimizer.step(oCommand);
    oTape = oCommand.bytes();

    // 5d. Optional standalone ZeroGrad Tape (zTape) if you want manual control
    CommandBuffer zCommand = CommandBuffer();
    optimizer.zeroGrad(zCommand);
    zTape = zCommand.bytes();
  }

  GPUTensor<dynamic> forward() {
    CudaEngine.run(fTape);
    return outputRef;
  }

  void zeroGrad() {
    // Used if you are accumulating gradients manually over multiple forward passes
    CudaEngine.run(zTape);
  }

  void backward() {
    // Note: In our compile method, bTape automatically zeros gradients
    // at the very start of the tape before executing the backpropagation.
    CudaEngine.run(bTape);
  }

  void optimize() {
    CudaEngine.run(oTape);
  }

  // --- 5. High-Level Execution (Developer Friendly) ---

  double trainStep(List<double> x, List<double> y) {
    // 1. Push raw floats directly into the pre-allocated VRAM addresses
    inputRef.pushData(x);
    targetRef.pushData(y);

    // 2. Fire the pre-compiled C++ execution instructions
    CudaEngine.run(fTape);
    CudaEngine.run(bTape); // (bTape handles zero_grad internally)
    CudaEngine.run(oTape);

    // 3. Sync only the scalar loss back to the Dart CPU heap
    lossRef.toCpu();
    return lossRef.value;
  }

  List<dynamic> predict(List<double> x) {
    // 1. Push raw floats to VRAM
    inputRef.pushData(x);

    // 2. Fire only the forward pass
    CudaEngine.run(fTape);

    // 3. Sync the final prediction tensor back to Dart
    outputRef.toCpu();
    return outputRef.value;
  }

  // --- 6. Memory Management ---

  void free() {
    // Free layer parameters (weights & biases)
    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].free();
    }

    // Free all temporary tensors generated during the forward pass
    for (int i = 0; i < intermediates.length; i = i + 1) {
      intermediates[i].free();
    }

    // Free the entry, exit, and loss tensors
    inputRef.free();
    targetRef.free();
    outputRef.free();
    lossRef.free();
  }
}