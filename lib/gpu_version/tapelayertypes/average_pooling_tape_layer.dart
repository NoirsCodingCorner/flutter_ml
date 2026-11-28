import '../ffi/allocation_buffer.dart';
import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Slides a two-dimensional window over a given matrix.
/// Each window returns the average over all its values.
class AveragePooling2DTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => "AveragePooling2DGPU";

  int poolSize;
  int stride;

  late int inputHeight;
  late int inputWidth;

  /// Slides a window over a given tensor in the first two dimensions with a size of [poolSize] and a stride of [stride].
  /// Each window returns the average over all its values.
  AveragePooling2DTL({this.poolSize = 2, this.stride = 2});

  /// Uses the inherent size of the [input] as its bounds. No additional allocation needs to be done.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    inputHeight = input.shape[0];
    inputWidth = input.shape[1];
    built = true;
  }

  /// Writes the [avgPool2dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out =
        avgPool2dGPU(input, poolSize, stride, tape, allocTape);
    return out;
  }

  /// This layer has no weights to return.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> emptyMap = <String, List<dynamic>>{};
    return emptyMap;
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}

/// Collapses a matrix input into a vector of the average values of its feature columns.
/// Does not store intermediate values.
class GlobalAveragePoolingTL extends TapeLayer<Matrix, Vector> {
  @override
  String get name => 'GlobalAveragePoolingGPU';

  GlobalAveragePoolingTL();

  /// No allocations or parameters are set in this layer.
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
    Map<String, List<dynamic>> emptyMap = <String, List<dynamic>>{};
    return emptyMap;
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}
