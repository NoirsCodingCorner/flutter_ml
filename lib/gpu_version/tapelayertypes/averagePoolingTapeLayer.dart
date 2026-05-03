import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class AveragePooling2DTL extends TapeLayer {
  int poolSize;
  int stride;

  late int inputHeight;
  late int inputWidth;

  AveragePooling2DTL({this.poolSize = 2, this.stride = 2});

  @override
  String get name {
    return 'AveragePooling2DGPU';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    inputHeight = input.shape[0];
    inputWidth = input.shape[1];
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    GPUTensor<Matrix> out = avgPool2dGPU(typedInput, poolSize, stride, tape);

    return out;
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> emptyMap = <String, List<dynamic>>{};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}

class GlobalAveragePoolingTL extends TapeLayer {

  GlobalAveragePoolingTL();

  @override
  String get name {
    return 'GlobalAveragePoolingGPU';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    GPUTensor<Vector> out = globalAveragePoolingGPU(typedInput, tape);

    return out;
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> emptyMap = <String, List<dynamic>>{};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}