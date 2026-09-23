import 'dart:typed_data';
import '../../logger.dart';
import '../full_library.dart';
import 'optimizer/adam.dart';

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

  void compile(List<int> inputShape, List<int> targetShape, double learningRate, {bool useAdam = true}) {
    // 1. Dummy Input/Target
    inputRef = GPUTensor<Matrix>.empty(inputShape);
    targetRef = GPUTensor<Matrix>.empty(targetShape);

    // 2. Build Layers
    GPUTensor<dynamic> current = inputRef;
    CommandBuffer buildTape = CommandBuffer();
    List<GPUTensor> buildTrash = <GPUTensor>[];

    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].build(current);
      current = layers[i].forward(current, buildTape, buildTrash);
    }
    for (int i = 0; i < buildTrash.length; i = i + 1) {
      buildTrash[i].free();
    }
    if (!buildTrash.contains(current)) current.free();

    // 3. Collect Params
    allParams.clear();
    for (int i = 0; i < layers.length; i = i + 1) {
      List<GPUTensor> params = layers[i].parameters;
      for (int j = 0; j < params.length; j = j + 1) {
        allParams.add(params[j]);
      }
    }

    // ✅ 4. OPTIMIZER WAHL
    if (useAdam) {
      optimizer = AdamGPU(allParams, learningRate);
    } else {
      optimizer = SGDGPU(allParams, learningRate);
    }

    // 5a. Trace Forward
    CommandBuffer fCommand = CommandBuffer();
    intermediates.clear();
    current = inputRef;
    for (int i = 0; i < layers.length; i = i + 1) {
      current = layers[i].forward(current, fCommand, intermediates);
    }
    outputRef = current;
    lossRef = binaryCrossEntropyGPU<Matrix>(outputRef as GPUTensor<Matrix>, targetRef, fCommand);
    fTape = fCommand.bytes();

    // 5b. Trace Backward
    CommandBuffer bCommand = CommandBuffer();
    lossRef.zeroGraphGrads(bCommand);
    lossRef.backward(bCommand, fillOnes: true);
    for (int i = 0; i < allParams.length; i = i + 1) {
      bCommand.putInt(OP_CLIP_GRAD_VALUE);
      bCommand.putString('${allParams[i].id}_grad');
      bCommand.putFloat(1.0);
    }
    bTape = bCommand.bytes();

    // 5c. Optional ZeroGrad Tape
    CommandBuffer zCommand = CommandBuffer();
    lossRef.zeroGraphGrads(zCommand);
    zTape = zCommand.bytes();
  }
  GPUTensor<dynamic> forward() {
    GPUEngine.run(fTape);
    return outputRef;
  }

  void zeroGrad() {
    // Used if you are accumulating gradients manually over multiple forward passes
    GPUEngine.run(zTape);
  }

  void backward() {
    // Note: In our compile method, bTape automatically zeros gradients
    // at the very start of the tape before executing the backpropagation.
    GPUEngine.run(bTape);
  }

  void optimize() {
    GPUEngine.run(oTape);
  }

  // --- 5. High-Level Execution (Developer Friendly) ---

  double trainStep(List<double> x, List<double> y) {
    inputRef.pushData(x);
    targetRef.pushData(y);

    CommandBuffer dynamicOCommand = CommandBuffer();
    optimizer.step(dynamicOCommand);
    oTape = dynamicOCommand.bytes();

    // ✅ FIX: Zwingend den VRAM-Müll der letzten Epoche löschen!
    GPUEngine.run(zTape);

    GPUEngine.run(fTape);
    GPUEngine.run(bTape);
    GPUEngine.run(oTape);

    lossRef.toCpu();
    return lossRef.value;
  }
  List<dynamic> predict(List<double> x) {
    // 1. Push raw floats to VRAM
    inputRef.pushData(x);

    // 2. Fire only the forward pass
    GPUEngine.run(fTape);

    // 3. Sync the final prediction tensor back to Dart
    outputRef.toCpu();
    return outputRef.value;
  }

  // --- 6. Memory Management ---

  void free() {
    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].free();
    }
    for (int i = 0; i < intermediates.length; i = i + 1) {
      intermediates[i].free();
    }
    inputRef.free();
    targetRef.free();
    lossRef.free();
    if (!intermediates.contains(outputRef)) {
      outputRef.free();
    }

    // Falls Adam genutzt wurde, die VRAM-Caches löschen!
    if (optimizer is AdamGPU) {
      (optimizer as AdamGPU).free();
    }
  }}