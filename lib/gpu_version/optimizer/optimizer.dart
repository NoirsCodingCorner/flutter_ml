/// Holds the optimizers for the GPU.
library GPUOptimizer;
export 'SGD.dart';


import '/tensor/tensor_gpu.dart';
import '../ffi/commandBuffer.dart';

/// Optimizers are used to update a parameters value to reduce the error of the system.
/// For that the gradient of the requested value is used.
/// Optimizers CAN be static so their execution tapes can be reused of they may require recompilation on every training step.
abstract class OptimizerGPU {
  /// List of parameters the gradient is responsible for updating.
  List<GPUTensor> parameters;

  OptimizerGPU(this.parameters);

  /// Function to add an optimization step to a deferred execution tape.
  void step(CommandBuffer tape);

  /// Function to reset the gradient of the parameter back to zero to make it ready for the next learning round.
  void zeroGrad(CommandBuffer tape);
}