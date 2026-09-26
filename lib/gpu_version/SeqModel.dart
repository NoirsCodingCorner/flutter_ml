import 'package:flutter_ml/full_library.dart';

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
  GPUTensor<Scalar> Function(GPUTensor<OutputType>, GPUTensor<OutputType>, CommandBuffer, {GPUTensor<Scalar>? outTensor})? lossFunction;

  /// A callback that builds the optimizer once all layer parameters are allocated in VRAM.
  OptimizerGPU Function(List<GPUTensor>)? optimizerBuilder;

  /// The optimizer responsible for updating layer weights based on calculated gradients.
  OptimizerGPU? optimizer;

  /// The compiled static command buffer for the forward pass during training.
  late CommandBuffer trainForward;

  /// The compiled static command buffer for backpropagation.
  late CommandBuffer trainBackward;

  /// The compiled static command buffer for weight updates and gradient clearing.
  late CommandBuffer trainOptimize;

  /// The compiled lightweight command buffer used exclusively for inference.
  CommandBuffer? inferTape;

  /// The entry point tensor mapped to the inference static graph.
  GPUTensor<InputType>? inferInput;

  /// The final output tensor produced by the [inferTape].
  GPUTensor<OutputType>? inferResult;

  /// Initializes the Sequential Model with a list of [layers] and the training [input] tensor.
  /// Optionally accepts a [target] tensor, [lossFunction], and an [optimizerBuilder] for training workflows.
  SeqModel(
      this.layers,
      this.input,
      {
        this.target,
        this.lossFunction,
        this.optimizerBuilder
      });

  /// Compiles the static training tapes for forward execution, backpropagation, and optimization.
  /// It allocates all layer parameters in VRAM, chains their forward passes, and records the gradient memory lifecycle.
  void compile() {
    trainForward = CommandBuffer();
    trainBackward = CommandBuffer();
    trainOptimize = CommandBuffer();

    GPUTensor<dynamic> current = input;
    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].build(current);
      current = layers[i].forward(current, trainForward, <GPUTensor>[]);
    }

    if (lossFunction != null && target != null) {
      loss = lossFunction!(current as GPUTensor<OutputType>, target!, trainForward);
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
      optimizer = optimizerBuilder!(allParams);
    }

    if (optimizer != null) {
      optimizer!.step(trainOptimize);
      optimizer!.zeroGrad(trainOptimize);
    }

    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].zeroStates(trainOptimize);
    }

    input.zeroGrad(trainOptimize);
    if (loss != null) {
      loss!.zeroGrad(trainOptimize);
    }
  }

  /// Compiles a lightweight static inference tape tailored to the shape of [newInferInput].
  /// This safely resizes internal layer VRAM caches (e.g., matching a batch size of 1) and returns the output tensor reference.
  GPUTensor<OutputType> predict(GPUTensor<InputType> newInferInput) {
    inferInput = newInferInput;
    inferTape = CommandBuffer();
    GPUTensor<dynamic> current = inferInput!;

    for (int i = 0; i < layers.length; i = i + 1) {
      current = layers[i].forward(current, inferTape!, <GPUTensor>[]);
    }

    inferResult = current as GPUTensor<OutputType>;
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

  /// Executes a single full training step consisting of the Forward, Backward, and Optimize tapes.
  /// Optionally accepts [inputData] and [targetData] to overwrite the respective VRAM buffers before execution.
  void runTraining({List<double>? inputData, List<double>? targetData}) {
    if (inputData != null) {
      input.pushData(inputData);
    }
    if (targetData != null && target != null) {
      target!.pushData(targetData);
    }

    GPUEngine.run(trainForward.bytes());
    GPUEngine.run(trainBackward.bytes());
    GPUEngine.run(trainOptimize.bytes());
  }

  /// Frees all VRAM allocations associated with the layers, including weights, biases, and intermediate caches.
  void free() {
    for (int i = 0; i < layers.length; i = i + 1) {
      layers[i].free();
    }
  }
}