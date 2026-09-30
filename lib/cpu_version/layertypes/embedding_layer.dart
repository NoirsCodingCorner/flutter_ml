import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/type_aliases.dart';
import '../layertypes/layer.dart';

/// Maps a 1D [Vector] tensor of integer indices (token IDs) to continuous embedding vectors, outputting a 2D [Matrix] tensor.
/// Operates as an embedding lookup table of shape `[vocabularySize, embeddingDimension]` on the CPU with dedicated autograd backpropagation.
class EmbeddingLayer extends Layer<Vector, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'embedding';

  /// Total number of unique tokens supported in the vocabulary.
  int vocabularySize;

  /// Dimensionality of each dense embedding vector.
  int embeddingDimension;

  /// Trainable embedding matrix tensor storing the lookup table.
  late Tensor<Matrix> embeddings;

  /// Creates an [EmbeddingLayer] with the given [vocabularySize] and [embeddingDimension].
  EmbeddingLayer(this.vocabularySize, this.embeddingDimension);

  /// Returns the trainable embedding table tensor.
  @override
  List<Tensor> get parameters => [embeddings];

  /// Allocates and initializes the embedding lookup table with uniform random values scaled by 0.01.
  @override
  void build(Tensor<Vector> input) {
    Random random = Random();
    // Initialize a flat list for the tensor data
    List<double> values = [];
    int totalElements = vocabularySize * embeddingDimension;

    for (int i = 0; i < totalElements; i = i + 1) {
      values.add((random.nextDouble() * 2 - 1) * 0.01);
    }

    // The Tensor constructor handles the Float32List conversion
    embeddings = Tensor<Matrix>(values);
    // Manually set shape since we passed a flat list
    embeddings.shape = [vocabularySize, embeddingDimension];

    super.build(input);
  }

  /// Performs the embedding lookup on the CPU for each token index in the input [Vector] and builds an autograd [Node] to route gradients to [embeddings].
  @override
  Tensor<Matrix> forward(Tensor<Vector> input) {
    int sequenceLength = input.shape[0];

    List<double> outputData = [];
    for (int i = 0; i < sequenceLength; i = i + 1) {
      int wordIndex = input.data[i].toInt();
      int embeddingOffset = wordIndex * embeddingDimension;

      for (int j = 0; j < embeddingDimension; j = j + 1) {
        outputData.add(embeddings.data[embeddingOffset + j]);
      }
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputData);
    out.shape = [sequenceLength, embeddingDimension];

    out.creator = Node([input, embeddings], () {
      for (int i = 0; i < sequenceLength; i = i + 1) {
        int wordIndex = input.data[i].toInt();
        int embOffset = wordIndex * embeddingDimension;
        int outOffset = i * embeddingDimension;

        for (int j = 0; j < embeddingDimension; j = j + 1) {
          embeddings.grad[embOffset + j] = embeddings.grad[embOffset + j] + out.grad[outOffset + j];
        }
      }
    }, opName: 'embedding_lookup');

    return out;
  }

  /// Returns the embedding table matrix as a map.
  @override
  Map<String, dynamic> getWeights() {
    return {'embeddings': embeddings.value};
  }

  /// Sets the embedding table matrix from a provided weights map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Matrix newWeights = weightsMap['embeddings'] as Matrix;
    for (int i = 0; i < vocabularySize; i = i + 1) {
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        embeddings.data[i * embeddingDimension + j] = newWeights[i][j];
      }
    }
  }
}

/// Maps a batched 2D [Matrix] tensor of integer indices `[batchSize, sequenceLength]` to dense continuous embeddings, outputting a 3D [Tensor3D] tensor.
/// Operates as a batched embedding lookup table of shape `[vocabularySize, embeddingDimension]` on the CPU with dedicated autograd backpropagation.
class EmbeddingLayerMatrix extends Layer<Matrix, Tensor3D> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'embedding_matrix';

  /// Total number of unique tokens supported in the vocabulary.
  int vocabularySize;

  /// Dimensionality of each dense embedding vector.
  int embeddingDimension;

  /// Trainable embedding matrix tensor storing the lookup table.
  late Tensor<Matrix> embeddings;

  /// Creates an [EmbeddingLayerMatrix] with the given [vocabularySize] and [embeddingDimension].
  EmbeddingLayerMatrix(this.vocabularySize, this.embeddingDimension);

  /// Returns the trainable embedding table tensor.
  @override
  List<Tensor> get parameters => [embeddings];

  /// Allocates and initializes the embedding lookup table with uniform random values scaled by 0.01.
  @override
  void build(Tensor<Matrix> input) {
    Random random = Random();
    List<double> values = [];
    int totalElements = vocabularySize * embeddingDimension;

    for (int i = 0; i < totalElements; i = i + 1) {
      values.add((random.nextDouble() * 2 - 1) * 0.01);
    }

    embeddings = Tensor<Matrix>(values);
    embeddings.shape = [vocabularySize, embeddingDimension];

    super.build(input);
  }

  /// Performs the batched embedding lookup on the CPU for all batch sequences and builds an autograd [Node] to accumulate gradients into [embeddings].
  @override
  Tensor<Tensor3D> forward(Tensor<Matrix> input) {
    int batchSize = input.shape[0];
    int sequenceLength = input.shape[1];

    List<double> outputData = [];
    for (int b = 0; b < batchSize; b = b + 1) {
      for (int s = 0; s < sequenceLength; s = s + 1) {
        int wordIndex = input.data[b * sequenceLength + s].toInt();
        int embOffset = wordIndex * embeddingDimension;

        for (int d = 0; d < embeddingDimension; d = d + 1) {
          outputData.add(embeddings.data[embOffset + d]);
        }
      }
    }

    Tensor<Tensor3D> out = Tensor<Tensor3D>(outputData);
    out.shape = [batchSize, sequenceLength, embeddingDimension];

    out.creator = Node([input, embeddings], () {
      int outSeqStride = sequenceLength * embeddingDimension;

      for (int b = 0; b < batchSize; b = b + 1) {
        for (int s = 0; s < sequenceLength; s = s + 1) {
          int wordIndex = input.data[b * sequenceLength + s].toInt();
          int embOffset = wordIndex * embeddingDimension;
          int outOffset = (b * outSeqStride) + (s * embeddingDimension);

          for (int d = 0; d < embeddingDimension; d = d + 1) {
            embeddings.grad[embOffset + d] = embeddings.grad[embOffset + d] + out.grad[outOffset + d];
          }
        }
      }
    }, opName: 'embedding_lookup_batch');

    return out;
  }

  /// Returns the embedding table matrix as a map.
  @override
  Map<String, dynamic> getWeights() {
    return {'embeddings': embeddings.value};
  }

  /// Sets the embedding table matrix from a provided weights map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Matrix newWeights = weightsMap['embeddings'] as Matrix;
    for (int i = 0; i < vocabularySize; i = i + 1) {
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        embeddings.data[i * embeddingDimension + j] = newWeights[i][j];
      }
    }
  }
}