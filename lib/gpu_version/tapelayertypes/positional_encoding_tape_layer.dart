import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

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

  /// Requires the maximum possible sequence length [maxLength] and the embedding dimension [dModel].
  PositionalEncodingTL(this.maxLength, this.dModel);

  /// Pre-calculates the static positional encoding matrix up to [maxLength] and uploads it to VRAM.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
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

    encodingMatrixTransposed = GPUTensor<Matrix>(
        peValuesTransposed, allocTape,
        sourceOp: '$name:encodingMatrixTransposed');
    built = true;
  }

  /// Appends the positional encoding addition to the [tape].
  /// Slices the pre-calculated matrix dynamically based on the current batch's sequence length.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    int sequenceLength = input.shape[0];

    GPUTensor<Matrix> slicedTransposed = sliceColumnGPU(
        encodingMatrixTransposed, 0, sequenceLength, tape, allocTape);

    GPUTensor<Matrix> positionalTensor =
    transposeGPU(slicedTransposed, tape, allocTape);

    GPUTensor<Matrix> out =
    addMatrixGPU(input, positionalTensor, tape, allocTape);

    return out;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}