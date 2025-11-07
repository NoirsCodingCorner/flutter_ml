import 'dart:io';
import 'dart:math';

import 'package:flutter_ml/optimizers/sgd.dart';

import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../diagnosysTools/logger.dart';
import '../layertypes/denseLayer.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';
import '../transformer/embeddingLayer.dart';
import '../transformer/globalpoolingLayer.dart';
import '../transformer/positionalEncodingLayer.dart';
import '../transformer/transformerEncodingLayer.dart';


Vector preprocessSentence(String sentence, Map<String, int> vocabulary, int maxLength) {
  String cleaned = sentence.toLowerCase().replaceAll(RegExp(r'[^\w\s]+'), '');
  List<double> tokens = [];
  for (String word in cleaned.split(' ')) {
    if (vocabulary.containsKey(word)) {
      tokens.add(vocabulary[word]!.toDouble());
    }
  }
  while (tokens.length < maxLength) {
    tokens.add(0.0);
  }
  return tokens.sublist(0, maxLength);
}

void main() {
  // --- 1. DATA LOADING ---
  String hemingwayText = File('lib/examples/hemingway.txt').readAsStringSync();
  String shakespeareText = File('lib/examples/shakespear.txt').readAsStringSync();
  Logger.cyan('Step 1: Loaded text files.', prefix: '✅');

  // --- 2. PREPROCESSING ---
  String allText = hemingwayText + " " + shakespeareText;
  String cleanedText = allText.toLowerCase().replaceAll(RegExp(r'[^\w\s]+'), '');
  Set<String> uniqueWords = Set<String>.from(cleanedText.split(RegExp(r'\s+')));

  Map<String, int> vocabulary = {'<pad>': 0};
  int index = 1;
  for (String word in uniqueWords) {
    if (word.isNotEmpty) {
      vocabulary[word] = index++;
    }
  }
  Logger.cyan('Step 2: Built vocabulary with ${vocabulary.length} unique words.', prefix: '✅');

  List<Map<String, dynamic>> dataset = [];
  int maxSequenceLength = 25;

  for (String sentence in hemingwayText.split(RegExp(r'[\.!?]'))) {
    if (sentence.trim().length > 5) {
      dataset.add({'text': sentence, 'label': 0.0}); // 0.0 for Hemingway
    }
  }
  for (String sentence in shakespeareText.split(RegExp(r'[\.!?]'))) {
    if (sentence.trim().length > 5) {
      dataset.add({'text': sentence, 'label': 1.0}); // 1.0 for Shakespeare
    }
  }
  Logger.cyan('Step 3: Created dataset with ${dataset.length} sentences.', prefix: '✅');

  // --- 3. TRAIN/TEST SPLIT ---
  dataset.shuffle();

  int splitIndex = (dataset.length * 0.8).floor();
  List<Map<String, dynamic>> trainData = dataset.sublist(0, splitIndex);
  List<Map<String, dynamic>> testData = dataset.sublist(splitIndex);

  List<Vector> trainInputs = [];
  List<Vector> trainTargets = [];
  for (Map<String, dynamic> item in trainData) {
    trainInputs.add(preprocessSentence(item['text'] as String, vocabulary, maxSequenceLength));
    trainTargets.add([item['label'] as double]);
  }

  List<Vector> testInputs = [];
  List<Vector> testTargets = [];
  for (Map<String, dynamic> item in testData) {
    testInputs.add(preprocessSentence(item['text'] as String, vocabulary, maxSequenceLength));
    testTargets.add([item['label'] as double]);
  }
  Logger.cyan('Step 4: Split data into ${trainInputs.length} training and ${testInputs.length} test samples.', prefix: '✅');

  // --- 4. MODEL DEFINITION & TRAINING ---
  int vocabSize = vocabulary.length;
  int dModel = 16;
  int numHeads = 4;
  int dff = 32;
  int numEncoderBlocks = 1;

  SNetwork styleClassifier = SNetwork([
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(maxSequenceLength, dModel),
    TransformerEncoderBlock(dModel, numHeads, dff),
    TransformerEncoderBlock(dModel, numHeads, dff),
    GlobalAveragePooling1D(),
    DenseLayer(1, activation: Sigmoid()),
  ]);

  styleClassifier.predict(Tensor<Vector>(trainInputs[0]));
  styleClassifier.compile(
      configuredOptimizer: Adam(styleClassifier.parameters, learningRate: 0.001)
  );

  styleClassifier.fit(trainInputs, trainTargets, epochs: 10, debug: true);

  // --- 5. EVALUATION ---
  Logger.green('\n--- FINAL EVALUATION ON UNSEEN TEST DATA ---', prefix: '📊');
  styleClassifier.evaluate(testInputs, testTargets);

  // --- 6. TESTING ON NEW SENTENCES ---
  Logger.blue('\n--- TESTING ON NEW SENTENCES ---', prefix: '🧪');
  List<String> testSentences = [
    "the old man and the sea",
    "wherefore art thou",
    "a farewell to arms",
    "o happy dagger this is thy sheath",
    "death which cannot choose"
  ];

  // Correct labels for the sentences above
  List<String> correctLabels = [
    "Hemingway",
    "Shakespeare",
    "Hemingway",
    "Shakespeare",
    "Hemingway"
  ];

  Tensor<Vector>? lastPrediction;

  for (int i = 0; i < testSentences.length; i++) {
    String sentence = testSentences[i];
    String correctLabel = correctLabels[i];

    Vector tokenized = preprocessSentence(sentence, vocabulary, maxSequenceLength);
    Tensor<Vector> inputTensor = Tensor<Vector>(tokenized);
    Tensor<Vector> prediction = styleClassifier.predict(inputTensor) as Tensor<Vector>;
    lastPrediction = prediction; // Save for graph printing

    // The model outputs > 0.5 for Shakespeare (label 1.0)
    String predictedLabel = (prediction.value[0] > 0.5) ? "Shakespeare" : "Hemingway";
    String resultEmoji = (predictedLabel == correctLabel) ? '✅ Correct' : '❌ Incorrect';

    print('Input: "$sentence"');
    print(' -> Prediction: $predictedLabel (Raw: ${prediction.value[0].toStringAsFixed(4)})');
    print(' -> Correct:    $correctLabel');
    print(' -> Result:     $resultEmoji\n');
  }

  // Print graph for the last sentence processed
  if (lastPrediction != null) {
    print('--- Computation Graph for last prediction ---');
    lastPrediction.printGraph();
    lastPrediction.printParallelGraph();
  }
}