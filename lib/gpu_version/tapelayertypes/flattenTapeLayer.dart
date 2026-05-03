import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class FlattenTL extends TapeLayer {
  @override
  String get name {
    return 'FlattenLayer';
  }

  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Tensor3D> typedInput = input as GPUTensor<Tensor3D>;
    GPUTensor<Matrix> out = flatten3DToMatrixGPU(typedInput, tape);
    return out;
  }

  @override
  void free() {
    // No weights to free
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    built = true;
  }
}