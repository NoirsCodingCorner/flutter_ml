import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/layer.dart';
import '../optimizers/adam.dart';
import '../optimizers/optimizers.dart';

/// A layer that turns a 1D vector of integer indices into a 2D matrix of dense vectors.
///
/// This layer is the standard first step for processing a single sequence of text
/// for a Natural Language Processing (NLP) task.
///
/// - **Input:** A `Tensor<Vector>` where the vector contains integer indices
///   representing a single sequence of words.
/// - **Output:** A `Tensor<Matrix>` representing the sequence of embedding
///   vectors, with a shape of `[sequence_length, embeddingDimension]`.
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
}

/// A layer that turns a 2D matrix of integer indices into a 3D tensor of dense vectors.
///
/// This layer is designed to process a **batch** of text sequences simultaneously.
///
/// - **Input:** A `Tensor<Matrix>` where each row is a sequence of integer
///   indices. Shape: `[batch_size, sequence_length]`.
/// - **Output:** A `Tensor<Tensor3D>` representing the batch of embedded
///   sequences. Shape: `[batch_size, sequence_length, embeddingDimension]`.
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
}

void main() {
  Map<String, int> vocabulary = {
    'king': 0, 'queen': 1, 'man': 2, 'woman': 3,
  };
  int vocabSize = vocabulary.length;
  int embeddingDimension = 2; // Use a 2D vector so we can visualize it

  List<List<dynamic>> trainingData = [
    ['king', 0.9],
    ['queen', 0.9],
    ['man', 0.1],
    ['woman', 0.1],
  ];

  EmbeddingLayer embeddingLayer = EmbeddingLayer(vocabSize, embeddingDimension);
  DenseLayer denseLayer = DenseLayer(1);

  // Build the layers with dummy data
  embeddingLayer.build(Tensor<Vector>([0]));
  denseLayer.build(Tensor<Vector>([0.0, 0.0]));

  List<Tensor> allParameters = [...embeddingLayer.parameters, ...denseLayer.parameters];
  Optimizer optimizer = Adam(allParameters, learningRate: 0.1);
  int epochs = 150;

  print('--- Starting Training ---');
  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0;
    for (List<dynamic> dataPoint in trainingData) {
      optimizer.zeroGrad();

      int wordIndex = vocabulary[dataPoint[0]]!;
      Tensor<Vector> input = Tensor<Vector>([wordIndex.toDouble()]);
      Tensor<Vector> target = Tensor<Vector>([dataPoint[1] as double]);

      Tensor<Matrix> embeddedSequence = embeddingLayer.call(input) as Tensor<Matrix>;
      Tensor<Vector> embeddedVector = selectRow(embeddedSequence, 0);
      Tensor<Vector> finalOutput = denseLayer.call(embeddedVector) as Tensor<Vector>;

      Tensor<Scalar> loss = mse(finalOutput, target);
      totalLoss += loss.value;

      loss.backward();
      optimizer.step();
    }
    if ((epoch + 1) % 15 == 0) {
      print('Epoch ${epoch + 1}, Avg Loss: ${totalLoss / trainingData.length}');
    }
  }
  print('--- Training Complete! ---\n');

  print('--- Proof of Learning ---');
  print('Final Learned Embedding Vectors:');
  vocabulary.forEach((word, index) {
    Vector vector = embeddingLayer.embeddings.value[index];
    print('  - $word: [${vector[0].toStringAsFixed(2)}, ${vector[1].toStringAsFixed(2)}]');
  });

  print('\nFinal Royalty Score Predictions:');
  for (List<dynamic> dataPoint in trainingData) {
    int wordIndex = vocabulary[dataPoint[0]]!;
    Tensor<Vector> input = Tensor<Vector>([wordIndex.toDouble()]);
    Tensor<Matrix> embeddedSequence = embeddingLayer.call(input) as Tensor<Matrix>;
    Tensor<Vector> embeddedVector = selectRow(embeddedSequence, 0);
    Tensor<Vector> finalOutput = denseLayer.call(embeddedVector) as Tensor<Vector>;
    print('  - Score for "${dataPoint[0]}": ${finalOutput.value[0].toStringAsFixed(4)} (Target: ${dataPoint[1]})');
  }
}