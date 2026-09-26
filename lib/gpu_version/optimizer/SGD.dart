import '../optimizer/optimizer.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';

/// Applies the SGD optimizer to every element of its parameter tensors.
/// SGD is a very simply descent method and my get stuck in local minima.
/// This function does not require to be reallocated on each training step and can be reused indefinitely.
class SGDGPU extends OptimizerGPU {
  /// The step size used to update the weights.
  double learningRate;

  /// Initializes the SGD optimizer, which does not allocate any additional VRAM.
  /// Requires the list of trainable [parameters] and the [learningRate].
  SGDGPU(super.parameters, this.learningRate);

  /// Appends the SGD weight update operations to the provided [tape].
  /// This optimizer does not require to be updated and can be reused indefinitely via the same tape execution.
  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_SGD_UPDATE);
      tape.putString(parameters[i].id);               // The weights
      tape.putString('${parameters[i].id}_grad');     // The gradients
      tape.putFloat(learningRate);
    }
  }

  /// Appends the zero-gradient operations to the [tape] to clear the gradients for the next pass.
  @override
  void zeroGrad(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${parameters[i].id}_grad');
    }
  }
}