import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Embedding layer to act as a discrete lookup table.
/// Processes a single sequence of indices (Vector) finding its embedding and returning a sequence of embeddings
/// as a matrix.
class EmbeddingTL extends TapeLayer<Vector, Matrix> {
  @override
  String get name => 'EmbeddingTapeLayer';

  int vocabularySize;
  int embeddingDimension;
  late GPUTensor<Matrix> embeddings;

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Matrix>? cachedOut;

  /// Requires the number of unique tokens in the dictionary [vocabularySize] as well as the
  /// number of embedding dimensions [embeddingDimension] used to represent each token.
  EmbeddingTL(this.vocabularySize, this.embeddingDimension);

  /// Allocates VRAM for [embeddings] with dimensionality: [vocabularySize]x[embeddingDimension].
  @override
  void build(GPUTensor<dynamic> input) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings = GPUTensor<Matrix>(values);
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
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Vector> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = embeddingLookupGPU(input, embeddings, tape, outTensor: cachedOut);
    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees the allocated [embeddings] and the cached output tensor.
  @override
  void free() {
    if (built) {
      embeddings.free();
    }
    if (cachedOut != null) {
      cachedOut!.free();
    }
  }

  /// Returns a map containing the [embeddings] of the layer as sole element.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.toCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  /// Sets the [embeddings] of the layer from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      embeddings.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    embeddings = GPUTensor<Matrix>(wData);
    built = true;
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

  /// --- Persistent Cache for Static Unrolling ---
  int cacheBatchSize = -1;
  GPUTensor<Tensor3D>? cachedOut;

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
  void build(GPUTensor<dynamic> input) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings = GPUTensor<Matrix>(values);
    built = true;
  }

  /// Writes the [embeddingLookupBatchGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Tensor3D> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = embeddingLookupBatchGPU(input, embeddings, tape, outTensor: cachedOut);
    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees the allocated [embeddings] and the cached output tensor.
  @override
  void free() {
    if (built) {
      embeddings.free();
    }
    if (cachedOut != null) {
      cachedOut!.free();
    }
  }

  /// Returns a map containing the [embeddings] of the layer as sole element.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.toCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  /// Sets the [embeddings] of the layer from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      embeddings.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    embeddings = GPUTensor<Matrix>(wData);
    built = true;
  }
}