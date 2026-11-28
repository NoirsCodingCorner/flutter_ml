import '/full_library.dart';

/// Sequential aid for constructing models to run.
/// The Sequential Model wraps a list of [TapeLayer]s to manage static graph compilation
/// via command tapes. It handles VRAM lifecycles, training loops, and high-speed inference.
class SeqModel<InputType, OutputType> {
  /// The ordered list of layers making up the sequential network.
  List<TapeLayer> layers;

  /// The entry point tensor used to trace the training graph.
  GPUTensor<InputType> input;

  /// The expected output tensor used for loss calculation during training.
  GPUTensor<OutputType>? target;

  /// The scalar tensor holding the computed loss value.
  GPUTensor<Scalar>? loss;

  /// The function used to calculate the loss between the network's output and the [target].
  GPUTensor<Scalar> Function(GPUTensor<OutputType>, GPUTensor<OutputType>,
      CommandBuffer, AllocBuffer)? lossFunction;

  /// A callback that builds the optimizer once all layer parameters are recorded in [allocTape].
  OptimizerGPU Function(List<GPUTensor>, AllocBuffer)? optimizerBuilder;

  /// The optimizer responsible for updating layer weights based on calculated gradients.
  OptimizerGPU? optimizer;

  /// The static allocation buffer collecting all layer weights, caches, and optimizer states.
  late AllocBuffer allocTape;

  /// The allocation buffer used for inference-specific intermediate caches.
  AllocBuffer? inferAllocTape;

  /// The compiled static command buffer for clearing all gradients before each training step.
  late CommandBuffer trainZeroGrad;

  /// The compiled static command buffer for the forward pass during training.
  late CommandBuffer trainForward;

  /// The compiled static command buffer for backpropagation.
  late CommandBuffer trainBackward;

  /// The compiled static command buffer for weight updates.
  late CommandBuffer trainOptimize;

  /// The compiled lightweight command buffer used exclusively for inference.
  CommandBuffer? inferTape;

  /// The entry point tensor mapped to the inference static graph.
  GPUTensor<InputType>? inferInput;

  /// The final output tensor produced by the [inferTape].
  GPUTensor<OutputType>? inferResult;

  /// Initializes the Sequential Model with a list of [layers] and the training [input] tensor.
  /// Optionally accepts a [target] tensor, [lossFunction], and an [optimizerBuilder] for training workflows.
  SeqModel(this.layers, this.input,
      {this.target, this.lossFunction, this.optimizerBuilder});

  /// Compiles the static training tapes for gradient clearing, forward execution, backpropagation, and optimization.
  /// Records all layer, loss, and optimizer allocations into [allocTape] and executes them once at the end.
  void compile() {
    allocTape = AllocBuffer();
    trainZeroGrad = CommandBuffer();
    trainForward = CommandBuffer();
    trainBackward = CommandBuffer();
    trainOptimize = CommandBuffer();

    GPUTensor<dynamic> current = input;
    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].build(current, allocTape);
      current = layers[i].forward(current, trainForward, allocTape);
    }

    if (lossFunction != null && target != null) {
      if (current is! GPUTensor<OutputType>) {
        throw StateError('[SeqModel Compile Error] Architecture Mismatch:\n'
            'The Sequential Model expects an output of type GPUTensor<$OutputType> to calculate the loss, '
            'but the final layer returned a ${current.runtimeType}.\n'
            'Please ensure the last TapeLayer matches your SeqModel definition.');
      }
      loss = lossFunction!(current, target!, trainForward, allocTape);
    }

    if (loss != null) {
      loss!.backward(trainBackward);
    }

    if (optimizerBuilder != null) {
      List<GPUTensor> allParams = <GPUTensor>[];
      for (int i = 0; i < layers.length; i = i + 1) {
        List<GPUTensor> layerParams = layers[i].parameters;
        for (int j = 0; j < layerParams.length; j = j + 1) {
          allParams.add(layerParams[j]);
        }
      }
      optimizer = optimizerBuilder!(allParams, allocTape);
    }

    if (optimizer != null) {
      optimizer!.step(trainOptimize);
    }

    allocTape.compileZeroGradTape(trainZeroGrad);
    input.zeroGrad(trainZeroGrad);

    allocTape.execute();
  }

  /// Compiles a lightweight static inference tape tailored to the shape of [newInferInput].
  /// Records any inference intermediates into [inferAllocTape] and executes them before returning the output tensor.
  GPUTensor<OutputType> predict(GPUTensor<InputType> newInferInput) {
    if (inferAllocTape != null) {
      inferAllocTape!.freeAll();
    }

    inferInput = newInferInput;
    inferAllocTape = AllocBuffer();
    inferTape = CommandBuffer();
    GPUTensor<dynamic> current = inferInput!;

    bool previousGrad = GPUTensor.globalGrad;
    GPUTensor.globalGrad = false;

    for (int i = 0; i < layers.length; i = i + 1) {
      current = layers[i].forward(current, inferTape!, inferAllocTape!);
    }

    GPUTensor.globalGrad = previousGrad;

    if (current is! GPUTensor<OutputType>) {
      throw StateError('[SeqModel Inference Error] Architecture Mismatch:\n'
          'The Sequential Model is defined to return a GPUTensor<$OutputType>, '
          'but the final layer produced a ${current.runtimeType}.');
    }

    inferTape!.patchBool('is_training', false);
    inferTape!.patchFloat('dropout_rate', 0.0);

    inferAllocTape!.execute();

    inferResult = current;
    return inferResult!;
  }

  /// Executes the pre-compiled inference tape.
  /// If [inputData] is provided, it pushes the new raw bytes directly into the VRAM of the inference tensor before running.
  void runForward({List<double>? inputData}) {
    if (inputData != null && inferInput != null) {
      inferInput!.pushData(inputData);
    }
    if (inferTape != null) {
      GPUEngine.run(inferTape!.bytes());
    }
  }

  /// Executes a single full training step consisting of the ZeroGrad, Forward, Backward, and Optimize tapes.
  /// Optionally accepts [inputData] and [targetData] to overwrite the respective VRAM buffers before execution.
  void runTraining({List<double>? inputData, List<double>? targetData}) {
    if (inputData != null) {
      input.pushData(inputData);
    }
    if (targetData != null && target != null) {
      target!.pushData(targetData);
    }

    trainForward.patchRandomInts('dropout_seed');
    optimizer?.refreshStep(trainOptimize);

    GPUEngine.run(trainZeroGrad.bytes());
    GPUEngine.run(trainForward.bytes());
    if (trainBackward.offset > 0) {
      GPUEngine.run(trainBackward.bytes());
    }
    if (trainOptimize.offset > 0) {
      GPUEngine.run(trainOptimize.bytes());
    }
  }

  /// Frees all VRAM allocations recorded in [allocTape] and [inferAllocTape].
  void free() {
    allocTape.freeAll();
    if (inferAllocTape != null) {
      inferAllocTape!.freeAll();
      inferAllocTape = null;
    }
  }
}