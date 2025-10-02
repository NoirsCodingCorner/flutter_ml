import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/layer.dart';
import '../optimizers/optimizers.dart';

/// A layer that turns positive integers (indices) into dense vectors of fixed size.
///
/// This layer is the standard first step for any Natural Language Processing (NLP)
/// task. It maps a vocabulary of a given size into a continuous vector space,
/// where the vectors are trainable parameters.
///
/// During training, the model learns to place words with similar meanings
/// close to each other in this vector space.
///
/// Following the framework's lifecycle, the internal embedding matrix is created
/// in the `build` method, which is called automatically the first time the layer
/// receives an input.
///
/// - **Input:** A `Tensor<Vector>` where the vector contains integer indices
///   representing a sequence of words.
/// - **Output:** A `Tensor<Matrix>` representing the sequence of embedding
///   vectors, with a shape of `[sequence_length, embeddingDimension]`.
///
/// ### Example
/// ```dart
/// Layer embedding = EmbeddingLayer(1000, 50);
///
/// Tensor<Vector> sentence = Tensor<Vector>([4, 25, 112]);
///
/// // The call below will first build the layer, then run the forward pass.
/// Tensor<Matrix> embeddedSentence = embedding.call(sentence) as Tensor<Matrix>;
/// ```
class EmbeddingLayer extends Layer {
  @override
  String name = 'embedding';
  int vocabularySize;
  int embeddingDimension;

  late Tensor<Matrix> embeddings;

  EmbeddingLayer(this.vocabularySize, this.embeddingDimension);

  @override
  List<Tensor> get parameters => [embeddings];

  /// Initializes the main `embeddings` matrix.
  ///
  /// Unlike a `DenseLayer`, this method does not need to inspect the input's
  /// shape, as its dimensions (`vocabularySize`, `embeddingDimension`) are
  /// known at instantiation. It initializes the matrix with small random values.
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

  /// Performs the forward pass by looking up the indices in the embedding matrix.
  ///
  /// For each integer index in the input vector, this method retrieves the
  /// corresponding vector from the `embeddings` matrix and returns the sequence
  /// of these vectors.
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
      // The backward pass is a "scatter add". For each word in the input sequence,
      // its corresponding gradient vector is added to the correct row of the
      // main embedding gradient matrix.
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

/// Selects a single row from a matrix and returns it as a vector.
Tensor<Vector> selectRow(Tensor<Matrix> m, int rowIndex) {
  Vector outValue = m.value[rowIndex];
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([m], () {
    // The backward pass adds the incoming gradient back to the correct row
    // of the original matrix's gradient.
    for (int i = 0; i < outValue.length; i++) {
      m.grad[rowIndex][i] += out.grad[i];
    }
  }, opName: 'selectRow', cost: 0);
  return out;
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