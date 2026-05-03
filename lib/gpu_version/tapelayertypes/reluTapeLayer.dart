import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class ReLULayerTL extends TapeLayer {
  @override
  String get name {
    return 'ReLULayerTapeLayer';
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

    // CORRECTED: Matches 'reluGPU' from tensor_math_gpu.dart
    GPUTensor<Vector> out = reluGPU(typedInput, tape);

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

class ReLULayerMatrixTapeLayer extends TapeLayer {
  @override
  String get name {
    return 'ReLULayerMatrixTapeLayer';
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

    GPUTensor<Matrix> out = reluMatrixGPU(typedInput, tape);

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