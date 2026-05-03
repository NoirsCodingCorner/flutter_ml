import '../optimizer/optimizer.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';

// Assuming your Optimizer abstract class is imported here

class SGDGPU extends OptimizerGPU {
  double learningRate;

  SGDGPU(super.parameters, this.learningRate);

  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_SGD_UPDATE);
      tape.putString(parameters[i].id);               // The weights
      tape.putString('${parameters[i].id}_grad');     // The gradients
      tape.putFloat(learningRate);
    }
  }

  @override
  void zeroGrad(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${parameters[i].id}_grad');
    }
  }
}