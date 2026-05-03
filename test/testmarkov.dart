import 'dart:math';

import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/tensor_math_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';


void main() {
  CudaEngine.initialize(debug: false);

  // 1. Our Training Corpus
  String text = "i love programming in dart and i love running code on the gpu because the gpu is very fast and i love fast code";

  List<String> words = text.split(" ");
  List<String> vocab = words.toSet().toList(); // Get unique words

  Map<String, int> w2i = { for (int i = 0; i < vocab.length; i++) vocab[i] : i };
  Map<int, String> i2w = { for (int i = 0; i < vocab.length; i++) i : vocab[i] };

  int numStates = vocab.length;
  int order = 2; // Look at the last 2 words to predict the next

  print("==========================================");
  print("[TEXT] Vocabulary Size: $numStates words");
  print("[TEXT] Markov Order: $order");
  print("==========================================\n");

  // 2. Encode text to integers for the GPU
  List<double> trainData = words.map((w) => w2i[w]!.toDouble()).toList();
  GPUTensor<Vector> sequence = GPUTensor<Vector>(trainData);

  // 3. Train the Markov Table (Build Tape & Execute)
  CommandBuffer buildTape = CommandBuffer();
  GPUTensor<Matrix> probTable = buildMarkovTableGPU(sequence, order, numStates, buildTape);

  print("[GPU] Training Markov Chain...");
  CudaEngine.run(buildTape.bytes());
  print("[GPU] Training Complete.\n");

  // 4. Setup the Auto-Regressive Decoding Graph
  // We start with the seed phrase: "i love"
  List<double> currentHistory = [w2i["i"]!.toDouble(), w2i["love"]!.toDouble()];
  GPUTensor<Matrix> historyBatch = GPUTensor<Matrix>(<List<double>>[currentHistory]);

  CommandBuffer predictTape = CommandBuffer();
  GPUTensor<Matrix> nextProbs = markovPredictGPU(historyBatch, probTable, numStates, predictTape);

  // We only compile the tape ONCE!
  var predictBytes = predictTape.bytes();

  // 5. The Generation Loop
  int generateLength = 15;
  List<String> generatedText = ["i", "love"];
  Random rand = Random();

  print("[DECODE] Generating $generateLength words...\n");

  for (int step = 0; step < generateLength; step++) {

    // Step A: Run the prediction tape
    CudaEngine.run(predictBytes);

    // Step B: Pull probabilities from VRAM to CPU
    nextProbs.toCpu();

    // Step C: Sample the next word based on the probability distribution
    double dartRand = rand.nextDouble(); // 0.0 to 1.0
    double cumulative = 0.0;
    int nextWordIndex = 0;

    for (int s = 0; s < numStates; s++) {
      cumulative += nextProbs.data[s]; // Output shape is [1, numStates]
      if (dartRand < cumulative) {
        nextWordIndex = s;
        break;
      }
    }

    String nextWord = i2w[nextWordIndex]!;
    generatedText.add(nextWord);

    // Step D: Shift the history window forward by 1
    currentHistory.removeAt(0); // Drop oldest word
    currentHistory.add(nextWordIndex.toDouble()); // Add newest word

    // Step E: Push the new history back to VRAM for the next loop!
    historyBatch.pushData(currentHistory);
  }

  print("==========================================");
  print("📝 GENERATED SEQUENCE:");
  print(generatedText.join(" "));
  print("==========================================");

  // Cleanup
  sequence.free();
  probTable.free();
  historyBatch.free();
  nextProbs.free();
  CudaEngine.dispose();
}