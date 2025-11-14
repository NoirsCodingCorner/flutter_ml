import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/layer.dart';
import '../optimizers/adam.dart';
import '../optimizers/optimizers.dart';

class EmbeddingLayer extends Layer {
  @override
  String name = 'embedding';
  int vocabularySize;
  int embeddingDimension;

  late Tensor<Matrix> embeddings;

  EmbeddingLayer(this.vocabularySize, this.embeddingDimension);

  @override
  List<Tensor> get parameters => [embeddings];

  @override
  void build(Tensor<dynamic> input) {
    Random random = Random();
    Matrix embeddingValues = [];
    for (int i = 0; i < vocabularySize; i++) {
      Vector row = [];
      for (int j = 0; j < embeddingDimension; j++) {
        row.add((random.nextDouble() * 2 - 1) * 0.01);
      }
      embeddingValues.add(row);
    }
    embeddings = Tensor<Matrix>(embeddingValues);
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Vector wordIndices = (input as Tensor<Vector>).value;
    Matrix outputSequence = [];

    for (double indexDouble in wordIndices) {
      int index = indexDouble.toInt();
      outputSequence.add(embeddings.value[index]);
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputSequence);

    out.creator = Node([embeddings], () {
      for (int i = 0; i < wordIndices.length; i++) {
        int index = wordIndices[i].toInt();
        for (int j = 0; j < embeddingDimension; j++) {
          embeddings.grad[index][j] += out.grad[i][j];
        }
      }
    }, opName: 'embedding_lookup', cost: 0);

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'embeddings': embeddings.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> embeddingsDynamic = weightsMap['embeddings'] as List<dynamic>;
    Matrix newEmbeddings = embeddingsDynamic.map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();

    for (int i = 0; i < embeddings.value.length; i++) {
      for (int j = 0; j < embeddings.value[0].length; j++) {
        embeddings.value[i][j] = newEmbeddings[i][j];
      }
    }
  }
}

class EmbeddingLayerMatrix extends Layer {
  @override
  String name = 'embedding_matrix';
  int vocabularySize;
  int embeddingDimension;

  late Tensor<Matrix> embeddings;

  EmbeddingLayerMatrix(this.vocabularySize, this.embeddingDimension);

  @override
  List<Tensor> get parameters => [embeddings];

  @override
  void build(Tensor<dynamic> input) {
    Random random = Random();
    Matrix embeddingValues = [];
    for (int i = 0; i < vocabularySize; i++) {
      Vector row = [];
      for (int j = 0; j < embeddingDimension; j++) {
        row.add((random.nextDouble() * 2 - 1) * 0.01);
      }
      embeddingValues.add(row);
    }
    embeddings = Tensor<Matrix>(embeddingValues);
    super.build(input);
  }

  @override
  Tensor<Tensor3D> forward(Tensor<dynamic> input) {
    Matrix batchIndices = (input as Tensor<Matrix>).value;
    Tensor3D outputBatch = [];

    for (Vector wordIndices in batchIndices) {
      Matrix outputSequence = [];
      for (double indexDouble in wordIndices) {
        int index = indexDouble.toInt();
        outputSequence.add(embeddings.value[index]);
      }
      outputBatch.add(outputSequence);
    }

    Tensor<Tensor3D> out = Tensor<Tensor3D>(outputBatch);

    out.creator = Node([embeddings], () {
      for (int b = 0; b < batchIndices.length; b++) {
        for (int i = 0; i < batchIndices[b].length; i++) {
          int index = batchIndices[b][i].toInt();
          for (int j = 0; j < embeddingDimension; j++) {
            embeddings.grad[index][j] += out.grad[b][i][j];
          }
        }
      }
    }, opName: 'embedding_lookup_batch', cost: 0);

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'embeddings': embeddings.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> embeddingsDynamic = weightsMap['embeddings'] as List<dynamic>;
    Matrix newEmbeddings = embeddingsDynamic.map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();

    for (int i = 0; i < embeddings.value.length; i++) {
      for (int j = 0; j < embeddings.value[0].length; j++) {
        embeddings.value[i][j] = newEmbeddings[i][j];
      }
    }
  }
}