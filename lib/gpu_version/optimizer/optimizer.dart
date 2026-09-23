/// Holds the optimizers for the GPU.
library GPUOptimizer;
export 'SGD.dart';


import '/tensor/tensor_gpu.dart';
import '../ffi/commandBuffer.dart';

abstract class OptimizerGPU {
  List<GPUTensor> parameters;

  OptimizerGPU(this.parameters);

  void step(CommandBuffer tape);

  void zeroGrad(CommandBuffer tape);
}