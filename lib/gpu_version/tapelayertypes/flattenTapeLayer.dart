import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Flattens a Tensor3D into a Matrix.
class FlattenTL extends TapeLayer<Tensor3D, Matrix> {
  @override
  String get name => 'FlattenLayer';

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Matrix>? cachedOut;

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

  /// Writes the [flatten3DToMatrixGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Tensor3D> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = flatten3DToMatrixGPU(input, tape, outTensor: cachedOut);
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
  void setWeights(Map<String, List<dynamic>> newWeights) {
    built = true;
  }
}