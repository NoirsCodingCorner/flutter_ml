import 'package:flutter_ml/optimizers/adam.dart';
import 'package:flutter_ml/transformer/positionalEncodingLayer.dart';
import 'package:flutter_ml/transformer/transformerEncodingLayer.dart';

import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../nets/snet.dart';
import 'embeddingLayer.dart';
import 'globalpoolingLayer.dart';

/*void main() {
  Map<String, int> vocabulary = {
    '<pad>': 0, 'i': 1, 'love': 2, 'this': 3, 'movie': 4, 'is': 5, 'great': 6,
    'a': 7, 'bad': 8, 'film': 9, 'hate': 10, 'terrible': 11, 'good': 12,
    'feel': 13, 'it': 14,
  };
  int vocabSize = vocabulary.length;

  List<List<dynamic>> trainingData = [
    ['i love this movie', 1.0], ['this film is great', 1.0],
    ['i hate this film', 0.0], ['this is a terrible movie', 0.0],
    ['this movie is good', 1.0], ['i feel great', 1.0],
    ['this is bad', 0.0], ['i hate it', 0.0],
  ];

  List<Vector> inputs = [];
  List<Vector> targets = [];
  for (List<dynamic> dataPoint in trainingData) {
    String sentence = dataPoint[0] as String;
    double label = dataPoint[1] as double;
    Vector tokenized = [];
    for(String word in sentence.split(' ')){
      tokenized.add(vocabulary[word]!.toDouble());
    }
    inputs.add(tokenized);
    targets.add([label]);
  }

  int dModel = 16;
  int numHeads = 2;
  int dff = 32;
  int numEncoderBlocks = 2;
  int maxSequenceLength = 10;

  SNetwork sentimentClassifier = SNetwork([
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(maxSequenceLength, dModel),
    TransformerEncoderBlock(dModel, numHeads, dff),
    TransformerEncoderBlock(dModel, numHeads, dff),
    GlobalAveragePooling1D(),
    DenseLayer(1, activation: Sigmoid()),
  ]);

  sentimentClassifier.predict(Tensor<Vector>([1, 2, 3]));
  sentimentClassifier.compile(
      configuredOptimizer: Adam(sentimentClassifier.parameters, learningRate: 0.01)
  );

  sentimentClassifier.fit(inputs, targets, epochs: 100,debug: true);

  print('\n--- FINAL EVALUATION ---');
  sentimentClassifier.evaluate(inputs, targets);

  print('\n--- TESTING ON UNSEEN SENTENCES ---');
  List<String> testSentences = [ 'this movie is great', 'i hate that film', 'great movie', ];
  for (String sentence in testSentences) {
    Vector tokenized = [];
    for(String word in sentence.split(' ')){
      if (vocabulary.containsKey(word)) {
        tokenized.add(vocabulary[word]!.toDouble());
      }
    }
    Tensor<Vector> inputTensor = Tensor<Vector>(tokenized);
    Tensor<Vector> prediction = sentimentClassifier.predict(inputTensor) as Tensor<Vector>;
    String result = (prediction.value[0] > 0.5) ? "Positive" : "Negative";
    print('Prediction for "$sentence": $result (Raw: ${prediction.value[0].toStringAsFixed(4)})');
    if(sentence==testSentences.last){
      prediction.printGraph();
    }
  }

}*/