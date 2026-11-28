import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// A dropout layer that randomly mutes some elements of the provided GPUTensor.
/// Works on Scalar, Vector, Matrix and Vector3D.
/// It scales the remaining active elements by 1 / (1 - rate).
class DropoutTL extends TapeLayer {
  /// The dropout probability for every element in the provided Tensor.
  double rate;

  /// Boolean flag to toggle between active and inactive. During training dropout is disabled.
  bool isTraining;

  @override
  String get name => 'DropoutTapeLayer';

  /// A dropout layer that randomly mutes some elements of the provided GPUTensor.
  /// Works on Scalar, Vector, Matrix and Vector3D.
  /// Requires the dropout probability rate [rate]. It assumes [isTraining] to be 'true' by default.
  DropoutTL(this.rate, {this.isTraining = true});

  /// Does not allocate any additional VRAM.
  @override
  void build(GPUTensor<dynamic> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [dropoutGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// When [isTraining] is set to false it passes the input Tensor as its output.
  @override
  GPUTensor<dynamic> forward(
      GPUTensor<dynamic> input, CommandBuffer tape, AllocBuffer allocTape) {
    if (isTraining == false || rate == 0.0) {
      return input;
    }

    GPUTensor<dynamic> out = dropoutGPU(input, rate, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> weights) {}
}
