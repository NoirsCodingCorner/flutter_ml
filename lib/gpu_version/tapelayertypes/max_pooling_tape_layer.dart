import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Slides a two-dimensional window over a given matrix.
/// Each window returns the maximum of all its values.
class MaxPooling2DTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'MaxPooling2DTapeLayer';

  int poolSize;
  int stride;

  /// Slides a window over a given tensor in the first two dimensions with a size of [poolSize] and a stride of [stride].
  /// Each window returns the maximum of all its values.
  MaxPooling2DTL({this.poolSize = 2, this.stride = 2});

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [maxPool2dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input,
      CommandBuffer tape,
      AllocBuffer allocTape,
      ) {
    GPUTensor<Matrix> out =
    maxPool2dGPU(input, poolSize, stride, tape, allocTape);
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

/// Slides a one-dimensional window over a given vector.
/// Each window returns the maximum of all its values.
class MaxPooling1DTL extends TapeLayer<Vector, Vector> {
  @override
  String get name => 'MaxPooling1DTapeLayer';

  int poolSize;
  int stride;

  /// Slides a window over a given vector with a size of [poolSize] and a stride of [stride].
  /// Each window returns the maximum of all its values.
  MaxPooling1DTL({this.poolSize = 2, this.stride = 2});

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<Vector> input, AllocBuffer allocTape) {
    built = true;
  }

  /// Writes the [maxPool1dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Vector> forward(
      GPUTensor<Vector> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Vector> out =
    maxPool1dGPU(input, poolSize, stride, tape, allocTape);
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