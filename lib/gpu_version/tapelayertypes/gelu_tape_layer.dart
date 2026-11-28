import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Applies the gelu function over a Vector.
class GeluLayerTL extends TapeLayer<Vector, Vector> {
  @override
  String get name => 'GeluLayerTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Vector> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [geluGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Vector> forward(
      GPUTensor<Vector> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Vector> out = geluGPU(input, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    return wMap;
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}

/// Applies the gelu function over a Matrix
class GeluLayerMatrixTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'GeluLayerMatrixTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [geluMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out = geluMatrixGPU(input, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    return wMap;
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}
