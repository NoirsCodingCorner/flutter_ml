import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class MaxPooling2DTL extends TapeLayer {
  int poolSize;
  int stride;

  MaxPooling2DTL({this.poolSize = 2, this.stride = 2});

  @override
  String get name {
    return 'MaxPooling2DTapeLayer';
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
    return maxPool2dGPU(typedInput, poolSize, stride, tape);
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

class MaxPooling1DTL extends TapeLayer {
  int poolSize;
  int stride;

  MaxPooling1DTL({this.poolSize = 2, this.stride = 2});

  @override
  String get name {
    return 'MaxPooling1DTapeLayer';
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
    GPUTensor<Vector> typedInput = input as GPUTensor<Vector>;
    return maxPool1dGPU(typedInput, poolSize, stride, tape);
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