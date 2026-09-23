import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class EmbeddingLayer extends Layer<Vector, Matrix> {
  @override
  String name = 'embedding';
  int vocabularySize;
  int embeddingDimension;

  late Tensor<Matrix> embeddings;

  EmbeddingLayer(this.vocabularySize, this.embeddingDimension);

  @override
  List<Tensor> get parameters => [embeddings];

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

  @override
  Tensor<Matrix> forward(Tensor<Vector> input) {
    int sequenceLength = input.shape[0];
    int totalOutputElements = sequenceLength * embeddingDimension;

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

  @override
  Map<String, dynamic> getWeights() {
    return {'embeddings': embeddings.value};
  }

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

class EmbeddingLayerMatrix extends Layer<Matrix, Tensor3D> {
  @override
  String name = 'embedding_matrix';
  int vocabularySize;
  int embeddingDimension;

  late Tensor<Matrix> embeddings;

  EmbeddingLayerMatrix(this.vocabularySize, this.embeddingDimension);

  @override
  List<Tensor> get parameters => [embeddings];

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

  @override
  Map<String, dynamic> getWeights() {
    return {'embeddings': embeddings.value};
  }

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