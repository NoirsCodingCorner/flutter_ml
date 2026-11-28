import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Collapses a matrix input to a vector containing the average values of its columns.
class GlobalAveragePooling1DTL extends TapeLayer<Matrix, Vector> {
  @override
  String get name => 'GlobalAveragePooling1DTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [globalAveragePoolingGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Vector> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Vector> out = globalAveragePoolingGPU(input, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}
