import 'tape_layer.dart';
import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_aliases.dart';
import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';

/// Applies the Sigmoid-function to each element in a matrix.
class SigmoidMatrixTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'SigmoidMatrixTapeLayer';

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [sigmoidMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out = sigmoidMatrixGPU(input, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() => <String, List<dynamic>>{};

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}
