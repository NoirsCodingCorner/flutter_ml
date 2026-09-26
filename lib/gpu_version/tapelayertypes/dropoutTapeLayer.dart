import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

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

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<dynamic>? cachedOut;

  /// A dropout layer that randomly mutes some elements of the provided GPUTensor.
  /// Works on Scalar, Vector, Matrix and Vector3D.
  /// Requires the dropout probability rate [rate]. It assumes [isTraining] to be 'true' by default.
  DropoutTL(this.rate, {this.isTraining = true});

  /// Returns an empty list since this layer has no learnable parameters.
  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  /// Does not allocate any additional VRAM.
  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  /// Writes the [dropoutGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  /// When [isTraining] is set to false it passes the input Tensor as its output.
  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    if (isTraining == false || rate == 0.0) {
      return input;
    }

    // Safely extract the batch size, defaulting to 1 for Scalars where shape is empty
    int currentBatchSize = input.shape.isEmpty ? 1 : input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = dropoutGPU(input, rate, tape, outTensor: cachedOut);
    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees the cached intermediate tensor.
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
  void setWeights(Map<String, List<dynamic>> weights) {}
}