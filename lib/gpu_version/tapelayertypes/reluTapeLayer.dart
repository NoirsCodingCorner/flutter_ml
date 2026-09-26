import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Applies the ReLu-function to each element in a vector.
class ReLULayerTL extends TapeLayer<Vector, Vector> {
  @override
  String get name => 'ReLULayerTapeLayer';

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Vector>? cachedOut;

  /// This layer has no parameters to adjust.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    return params;
  }

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  /// Writes the [reluGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
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

    cachedOut = reluGPU(input, tape, outTensor: cachedOut);
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

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Matrix>? cachedOut;

  /// This layer has no parameters to adjust.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    return params;
  }

  /// Does not allocate additional VRAM.
  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  /// Writes the [reluMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = reluMatrixGPU(input, tape, outTensor: cachedOut);
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
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    return wMap;
  }

  /// This layer has no weights to set.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}