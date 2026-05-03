import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class GlobalAveragePooling1DTL extends TapeLayer {
  @override
  String get name {
    return 'GlobalAveragePooling1DTapeLayer';
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
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    return globalAveragePoolingGPU(typedInput, tape);
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}