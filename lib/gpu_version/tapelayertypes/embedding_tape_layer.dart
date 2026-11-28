import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Embedding layer to act as a discrete lookup table.
/// Processes a single sequence of indices (Vector) finding its embedding and returning a sequence of embeddings
/// as a matrix.
class EmbeddingTL extends TapeLayer<Vector, Matrix> {
  @override
  String get name => 'EmbeddingTapeLayer';

  int vocabularySize;
  int embeddingDimension;
  late GPUTensor<Matrix> embeddings;

  /// Requires the number of unique tokens in the dictionary [vocabularySize] as well as the
  /// number of embedding dimensions [embeddingDimension] used to represent each token.
  EmbeddingTL(this.vocabularySize, this.embeddingDimension);

  /// Allocates VRAM for [embeddings] with dimensionality: [vocabularySize]x[embeddingDimension].
  @override
  void build(GPUTensor<Vector> input, AllocBuffer allocTape) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings =
        GPUTensor<Matrix>(values, allocTape, sourceOp: '$name:embeddings');
    built = true;
  }

  /// Returns a list with the only element being the [embeddings].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(embeddings);
    }
    return params;
  }

  /// Writes the [embeddingLookupGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Vector> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out =
        embeddingLookupGPU(input, embeddings, tape, allocTape);
    return out;
  }

  /// Returns a map containing the [embeddings] of the layer as sole element.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.dataToCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  /// Sets the [embeddings] of the layer from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> flatWeights = <double>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        flatWeights.add(rawRow[j] as double);
      }
    }

    embeddings.pushData(flatWeights);
  }
}

/// Embedding layer to act as a discrete lookup table.
/// Processes a batch of sequences of indices (Matrix) finding its embedding and returning a batch of sequences of embeddings
/// as a Tensor3D.
class EmbeddingMatrixTL extends TapeLayer<Matrix, Tensor3D> {
  @override
  String get name => 'EmbeddingMatrixTapeLayer';

  int vocabularySize;
  int embeddingDimension;

  late GPUTensor<Matrix> embeddings;

  /// Requires the number of unique tokens in the dictionary [vocabularySize] as well as the
  /// number of embedding dimensions [embeddingDimension] used to represent each token.
  EmbeddingMatrixTL(this.vocabularySize, this.embeddingDimension);

  /// Returns a list with the only element being the [embeddings].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(embeddings);
    }
    return params;
  }

  /// Allocates VRAM for [embeddings] with dimensionality: [vocabularySize]x[embeddingDimension].
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings =
        GPUTensor<Matrix>(values, allocTape, sourceOp: '$name:embeddings');
    built = true;
  }

  /// Writes the [embeddingLookupBatchGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  @override
  GPUTensor<Tensor3D> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Tensor3D> out =
        embeddingLookupBatchGPU(input, embeddings, tape, allocTape);
    return out;
  }

  /// Returns a map containing the [embeddings] of the layer as sole element.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.dataToCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  /// Sets the [embeddings] of the layer from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> flatWeights = <double>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        flatWeights.add(rawRow[j] as double);
      }
    }

    embeddings.pushData(flatWeights);
  }
}
