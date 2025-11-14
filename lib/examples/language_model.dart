import 'dart:io';
import 'dart:math';
import 'dart:convert'; // For stdin/stdout

// Core imports
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../layertypes/layer.dart';

// Optimizers
import 'package:flutter_ml/optimizers/adam.dart';
import 'package:flutter_ml/optimizers/optimizers.dart';

// Layer imports
import '../transformer/embeddingLayer.dart';
import 'package:flutter_ml/transformer/positionalEncodingLayer.dart';
import 'package:flutter_ml/transformer/transformerEncodingLayer.dart';
import '../layertypes/denseLayer.dart'; // We need DenseLayerMatrix
import '../activationFunctions/softmax.dart'; // Import softmax

// ---
// A Generative Language Model trained on "shakespear.txt"
// ---
Future<void> main() async {
  // --- 1. File Loading ---
  print('💾 Loading shakespear.txt...');
  // Assuming the path is relative to where you run dart
  File file = File('lib/examples/shakespear.txt');
  if (!await file.exists()) {
    print('Error: shakespear.txt not found. Make sure it is in the same directory.');
    return;
  }
  String text = await file.readAsString();

  // --- 2. Vocabulary Building ---
  print('📚 Building vocabulary...');
  String normalized = text.toLowerCase().replaceAll(RegExp(r'[^a-z ]'), ' ');
  List<String> words = normalized.split(' ').where((s) => s.isNotEmpty).toList();

  Set<String> uniqueWords = Set<String>.from(words);
  Map<String, int> vocabulary = {'<pad>': 0};
  Map<int, String> vocabReversed = {0: '<pad>'};
  int index = 1;
  for (String word in uniqueWords) {
    vocabulary[word] = index;
    vocabReversed[index] = word;
    index++;
  }
  int vocabSize = vocabulary.length;
  print('Vocabulary size: $vocabSize words.');

  // --- 3. Data Preprocessing (Sliding Window) ---
  print('🛠️  Preprocessing data...');
  List<double> allTokens = words.map((w) => vocabulary[w]!.toDouble()).toList();
  List<Vector> inputs = [];
  List<Vector> targets = [];

  int sequenceLength = 20;

  for (int i = 0; i < allTokens.length - sequenceLength - 1; i++) {
    inputs.add(allTokens.sublist(i, i + sequenceLength));
    targets.add(allTokens.sublist(i + 1, i + sequenceLength + 1));
  }
  print('Created ${inputs.length} training samples.');

  if (inputs.isEmpty) {
    print('No training samples created. Is the text file empty or too short?');
    return;
  }

  // --- 4. Model Definition ---
  int dModel = 64;
  int numHeads = 4;
  int dff = 128;
  int maxSequenceLength = sequenceLength + 10;

  SNetwork model = SNetwork([
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(maxSequenceLength, dModel),
    TransformerEncoderBlock(dModel, numHeads, dff),
    TransformerEncoderBlock(dModel, numHeads, dff),
    DenseLayerMatrix(vocabSize),
  ]);

  // --- 5. Build and Compile ---
  Tensor<Vector> dummyInput = Tensor<Vector>(inputs[0]);
  model.predict(dummyInput); // Build the model
  Adam optimizer = Adam(model.parameters, learningRate: 0.01);

  // --- 6. Custom Training Loop ---
  print('--- STARTING TRAINING FOR 1 EPOCH ---');
  int epochs = 1; // *** MODIFIED: Run for only 1 epoch ***

  for (int epoch = 0; epoch < epochs; epoch++) {
    double epochLoss = 0;
    int samplesToTrain = min(1000, inputs.length);

    for (int i = 0; i < samplesToTrain; i++) {
      optimizer.zeroGrad();

      Tensor<Vector> inputTensor = Tensor<Vector>(inputs[i]);
      Tensor<Vector> targetTensor = Tensor<Vector>(targets[i]);

      Tensor<Matrix> predictionMatrix = model.call(inputTensor) as Tensor<Matrix>;
      Tensor<Scalar> loss = softmaxCrossEntropyLoss(predictionMatrix, targetTensor);
      epochLoss += loss.value;

      loss.backward();
      optimizer.step();

      // *** ADDED: Print graph for the last sample ***
      if (i == samplesToTrain - 1) {
        print('\n--- 📊 COMPUTATIONAL GRAPH (last sample) ---');
        loss.printGraph();
        print('-------------------------------------------\n');
      }
    }
    print('Epoch ${epoch + 1}/$epochs, Avg Loss: ${epochLoss / samplesToTrain}');
  }
  print('--- TRAINING FINISHED ---');

  // --- 7. Save the Model ---
  String modelPath = 'language_model_epoch_1.json';
  print('\n--- 💾 SAVING MODEL ---');
  await model.save(modelPath);
  print('Model saved to $modelPath');

  // --- 8. Removed Interactive Console Loop ---
  print('\nProcess finished.');
}

// ---
// --- HELPER FUNCTIONS (Unchanged) ---
// ---

List<double> tokenize(String sentence, Map<String, int> vocabulary) {
  List<double> tokens = [];
  String normalized = sentence.toLowerCase().replaceAll(RegExp(r'[^a-z ]'), ' ');
  List<String> words = normalized.split(' ').where((s) => s.isNotEmpty).toList();

  for (String word in words) {
    if (vocabulary.containsKey(word)) {
      tokens.add(vocabulary[word]!.toDouble());
    } else {
      tokens.add(vocabulary['<pad>']!.toDouble());
    }
  }
  return tokens;
}

String detokenize(List<double> tokens, Map<int, String> vocabReversed) {
  List<String> words = [];
  for (double tokenDouble in tokens) {
    int token = tokenDouble.toInt();
    if (vocabReversed.containsKey(token)) {
      String word = vocabReversed[token]!;
      if (word != '<pad>') {
        words.add(word);
      }
    }
  }
  return words.join(' ');
}

int argmax(Vector v) {
  int maxIndex = 0;
  double maxValue = v[0];
  for (int i = 1; i < v.length; i++) {
    if (v[i] > maxValue) {
      maxValue = v[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

String generateText(
    SNetwork model,
    Map<String, int> vocabulary,
    Map<int, String> vocabReversed,
    String seedText, {
      int length = 3,
    }) {
  List<double> tokens = tokenize(seedText, vocabulary);
  List<double> generatedTokens = [];

  for (int i = 0; i < length; i++) {
    Tensor<Vector> inputTensor = Tensor<Vector>(tokens);
    Tensor<Matrix> predictionMatrix = model.predict(inputTensor) as Tensor<Matrix>;
    Vector lastWordLogits = predictionMatrix.value[tokens.length - 1];
    int predictedIndex = argmax(lastWordLogits);
    if (predictedIndex == vocabulary['<pad>']) {
      break;
    }
    tokens.add(predictedIndex.toDouble());
    generatedTokens.add(predictedIndex.toDouble());
  }
  return detokenize(generatedTokens, vocabReversed);
}

Tensor<Scalar> softmaxCrossEntropyLoss(Tensor<Matrix> predictions, Tensor<Vector> targets) {
  Tensor<Matrix> probabilities = softmaxMatrix(predictions);

  int sequenceLength = targets.value.length;
  int vocabSize = probabilities.value[0].length;
  double totalLoss = 0.0;

  for (int i = 0; i < sequenceLength; i++) {
    int targetIndex = targets.value[i].toInt();
    if (targetIndex >= vocabSize) continue;
    double correctProbability = probabilities.value[i][targetIndex];
    totalLoss += -log(correctProbability + 1e-9);
  }

  double avgLoss = totalLoss / sequenceLength.toDouble();
  Tensor<Scalar> out = Tensor<Scalar>(avgLoss);

  out.creator = Node([predictions], () {
    for (int i = 0; i < sequenceLength; i++) {
      int targetIndex = targets.value[i].toInt();
      if (targetIndex >= vocabSize) continue;
      for (int j = 0; j < vocabSize; j++) {
        double grad = probabilities.value[i][j];
        if (j == targetIndex) {
          grad -= 1;
        }
        predictions.grad[i][j] += grad / sequenceLength.toDouble();
      }
    }
  }, opName: 'softmax_ce_loss');

  return out;
}