import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_aliases.dart';
import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Flattens a Tensor3D into a Matrix.
class FlattenTL extends TapeLayer<Tensor3D, Matrix> {
  @override
  String get name => 'FlattenLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Tensor3D> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [flatten3DToMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Tensor3D> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out = flatten3DToMatrixGPU(input, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    built = true;
  }
}
