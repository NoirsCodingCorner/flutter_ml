/// Holds the optimizers for the GPU.
library;

export 'sgd.dart';
export 'adam.dart';

import '/tensor/tensor_gpu.dart';
import '../ffi/command_buffer.dart';

/// Optimizers are used to update a parameters value to reduce the error of the system.
/// For that the gradient of the requested value is used.
/// Optimizers compile their update instructions into a reusable execution tape and patch dynamic values per step.
abstract class OptimizerGPU {
  /// List of parameters the optimizer is responsible for updating.
  List<GPUTensor> parameters;
  double learningRate;

  OptimizerGPU(this.parameters, this.learningRate);

  /// Function to add an optimization step to a deferred execution tape.
  void step(CommandBuffer tape);

  /// Called once before every replay of an already-compiled tape to patch dynamic values such as [learningRate] or step count.
  void refreshStep(CommandBuffer tape) {
    tape.patchFloat('lr', learningRate);
  }
}