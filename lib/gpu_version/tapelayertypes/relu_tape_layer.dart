import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Applies the ReLu-function to each element in a vector.
class ReLULayerTL extends TapeLayer<Vector, Vector> {
  @override
  String get name => 'ReLULayerTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Vector> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [reluGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Vector> forward(
      GPUTensor<Vector> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Vector> out = reluGPU(input, tape, allocTape);
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

/// Applies the ReLu-function to each element in a matrix.
class ReLULayerMatrixTapeLayer extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'ReLULayerMatrixTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [reluMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out = reluMatrixGPU(input, tape, allocTape);
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