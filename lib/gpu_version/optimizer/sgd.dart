import '/tensor/tensor_math_gpu.dart';
import 'optimizer.dart';
import '../ffi/command_buffer.dart';

/// Applies the SGD optimizer to every element of its parameter tensors.
/// SGD is a very simply descent method and my get stuck in local minima.
/// This function does not require to be reallocated on each training step and can be reused indefinitely.
class SGDGPU extends OptimizerGPU {
  /// Initializes the SGD optimizer, which does not allocate any additional VRAM.
  /// Requires the list of trainable [parameters] and the [learningRate].
  SGDGPU(super.parameters, super.learningRate);

  /// Appends the SGD weight update operations to the provided [tape].
  /// This optimizer does not require to be updated and can be reused indefinitely via the same tape execution.
  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      sgdUpdateGPU(parameters[i], learningRate, tape);
    }
  }
}
