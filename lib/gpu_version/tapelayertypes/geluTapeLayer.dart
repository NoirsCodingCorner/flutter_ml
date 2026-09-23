import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class GeluLayerTL extends TapeLayer {
  @override
  String get name {
    return 'GeluLayerTapeLayer';
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
    GPUTensor<Vector> typedInput = input as GPUTensor<Vector>;

    GPUTensor<Vector> out = geluGPU(typedInput, tape);

    intermediates.add(out);
    return out;
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}

class GeluLayerMatrixTL extends TapeLayer {
  @override
  String get name {
    return 'GeluLayerMatrixTapeLayer';
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

    GPUTensor<Matrix> out = geluMatrixGPU(typedInput, tape);

    intermediates.add(out);
    return out;
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}