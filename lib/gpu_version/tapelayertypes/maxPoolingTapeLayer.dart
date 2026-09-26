import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Slides a two-dimensional window over a given matrix.
/// Each window returns the maximum of all its values.
class MaxPooling2DTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'MaxPooling2DTapeLayer';

  int poolSize;
  int stride;

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Matrix>? cachedOut;

  /// Slides a window over a given tensor in the first two dimensions with a size of [poolSize] and a stride of [stride].
  /// Each window returns the maximum of all its values.
  MaxPooling2DTL({this.poolSize = 2, this.stride = 2});

  /// This layer has no parameters to adjust.
  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  /// Writes the [maxPool2dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input,
      CommandBuffer tape,
      List<GPUTensor> intermediates,
      ) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = maxPool2dGPU(input, poolSize, stride, tape, outTensor: cachedOut);
    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees the cached output tensor.
  @override
  void free() {
    if (cachedOut != null) {
      cachedOut!.free();
    }
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

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Vector>? cachedOut;

  /// Slides a window over a given vector with a size of [poolSize] and a stride of [stride].
  /// Each window returns the maximum of all its values.
  MaxPooling1DTL({this.poolSize = 2, this.stride = 2});

  /// This layer has no parameters to adjust.
  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  /// Writes the [maxPool1dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Vector> forward(GPUTensor<Vector> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape.isEmpty ? 1 : input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = maxPool1dGPU(input, poolSize, stride, tape, outTensor: cachedOut);
    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees the cached output tensor.
  @override
  void free() {
    if (cachedOut != null) {
      cachedOut!.free();
    }
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