import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Injects spatial/sequential context into tokens. Because self-attention is permutation-invariant,
/// this layer adds fixed sine and cosine waves of different frequencies to the embeddings,
/// allowing the model to perceive the absolute and relative positions of words in a sequence.
class PositionalEncodingTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'PositionalEncodingTapeLayer';

  int maxLength;
  int dModel;

  /// A non-trainable matrix storing the pre-calculated sine and cosine waves.
  /// Stored transposed to optimize memory access during slicing.
  late GPUTensor<Matrix> encodingMatrixTransposed;

  /// Persistent Cache for Static Unrolling
  int cacheSeqLength = -1;
  final List<GPUTensor<Matrix>> stepCache = <GPUTensor<Matrix>>[];

  /// Requires the maximum possible sequence length [maxLength] and the embedding dimension [dModel].
  PositionalEncodingTL(this.maxLength, this.dModel);

  /// This layer has no learnable parameters.
  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  /// Pre-calculates the static positional encoding matrix up to [maxLength] and uploads it to VRAM.
  @override
  void build(GPUTensor<Matrix> input) {
    List<List<double>> peValuesTransposed = <List<double>>[];
    for (int i = 0; i < dModel; i = i + 1) {
      List<double> row = <double>[];
      for (int pos = 0; pos < maxLength; pos = pos + 1) {
        double angle = pos / pow(10000, (2 * i) / dModel);
        if (i % 2 == 0) {
          row.add(sin(angle));
        } else {
          row.add(cos(angle));
        }
      }
      peValuesTransposed.add(row);
    }

    encodingMatrixTransposed = GPUTensor<Matrix>(peValuesTransposed);
    built = true;
  }

  /// Appends the positional encoding addition to the [tape].
  /// Slices the pre-calculated matrix dynamically based on the current batch's sequence length.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int sequenceLength = input.shape[0];

    bool useCache = (cacheSeqLength == sequenceLength);

    if (!useCache) {
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = sequenceLength;
    }

    int cIdx = 0;

    T? getCached<T>() {
      if (useCache) {
        T cached = stepCache[cIdx] as T;
        cIdx = cIdx + 1;
        return cached;
      }
      return null;
    }

    void saveCached(GPUTensor<Matrix> tensor) {
      if (!useCache) {
        stepCache.add(tensor);
      }
    }

    GPUTensor<Matrix> slicedTransposed = sliceColumnGPU(encodingMatrixTransposed, 0, sequenceLength, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(slicedTransposed);

    GPUTensor<Matrix> positionalTensor = transposeGPU(slicedTransposed, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(positionalTensor);

    GPUTensor<Matrix> out = addMatrixGPU(input, positionalTensor, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(out);

    return out;
  }

  /// Clears the gradients of all persistently cached intermediate tensors.
  @override
  void zeroStates(CommandBuffer tape) {
    for (int i = 0; i < stepCache.length; i = i + 1) {
      stepCache[i].zeroGrad(tape);
    }
  }

  /// Frees VRAM for the pre-calculated encoding matrix and persistently cached intermediate tensors.
  @override
  void free() {
    if (built) {
      encodingMatrixTransposed.free();

      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = -1;
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}