import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Collapses a matrix input to a vector containing the average values of its columns.
class GlobalAveragePooling1DTL extends TapeLayer<Matrix, Vector> {
  @override
  String get name => 'GlobalAveragePooling1DTapeLayer';

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Vector>? cachedOut;

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

  /// Writes the [globalAveragePoolingGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Vector> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = globalAveragePoolingGPU(input, tape, outTensor: cachedOut);
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