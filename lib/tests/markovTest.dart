import 'dart:math';
import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/tensor_math_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';

void main() async {
  print("🚀 Starting E2E GPU Test: Markov Chain...");

  // 1. Arrange: Initialize Engine
  await CudaEngine.initialize(debug: false);

  // Variables declared outside so they can be safely freed in the `finally` block
  GPUTensor<Vector>? sequence;
  GPUTensor<Matrix>? probTable;
  GPUTensor<Matrix>? historyBatch;
  GPUTensor<Matrix>? nextProbs;

  try {
    // 2. Arrange: Prepare Data
    String text = "i love programming in dart and i love running code on the gpu because the gpu is very fast and i love fast code";
    List<String> words = text.split(" ");
    List<String> vocab = words.toSet().toList();

    Map<String, int> w2i = { for (int i = 0; i < vocab.length; i++) vocab[i] : i };
    Map<int, String> i2w = { for (int i = 0; i < vocab.length; i++) i : vocab[i] };

    int numStates = vocab.length;
    int order = 2;

    List<double> trainData = words.map((w) => w2i[w]!.toDouble()).toList();
    sequence = GPUTensor<Vector>(trainData);

    // 3. Act: Train Model
    CommandBuffer buildTape = CommandBuffer();
    probTable = buildMarkovTableGPU(sequence, order, numStates, buildTape);
    CudaEngine.run(buildTape.bytes());

    // 4. Act: Prepare Generation
    List<double> currentHistory = [w2i["i"]!.toDouble(), w2i["love"]!.toDouble()];
    historyBatch = GPUTensor<Matrix>(<List<double>>[currentHistory]);

    CommandBuffer predictTape = CommandBuffer();
    nextProbs = markovPredictGPU(historyBatch, probTable, numStates, predictTape);
    var predictBytes = predictTape.bytes();

    int generateLength = 15;
    List<String> generatedText = ["i", "love"];

    // IMPORTANT: Fix the random seed to '42' so the test is deterministic!
    // This ensures it generates the exact same sequence every time you run it.
    Random rand = Random(42);

    for (int step = 0; step < generateLength; step++) {
      CudaEngine.run(predictBytes);
      nextProbs.toCpu();

      double dartRand = rand.nextDouble();
      double cumulative = 0.0;
      int nextWordIndex = 0;

      for (int s = 0; s < numStates; s++) {
        cumulative += nextProbs.data[s];
        if (dartRand < cumulative) {
          nextWordIndex = s;
          break;
        }
      }

      String nextWord = i2w[nextWordIndex]!;
      generatedText.add(nextWord);

      currentHistory.removeAt(0);
      currentHistory.add(nextWordIndex.toDouble());
      historyBatch.pushData(currentHistory);
    }

    // 5. Assert: Verify the Output

    // Check 1: Length must match seed + generated steps (2 + 15 = 17)
    if (generatedText.length != 17) {
      throw Exception("❌ TEST FAILED: Expected 17 words, got ${generatedText.length}.");
    }

    // Check 2: Verify every generated word actually belongs to the corpus vocabulary
    for (String word in generatedText) {
      if (!vocab.contains(word)) {
        throw Exception("❌ TEST FAILED: Generated invalid word '$word' not present in training corpus.");
      }
    }

    // Check 3: Deterministic Sequence Check (Random(42) output)
    List<String> expectedSequence = [
      "i", "love", "programming", "in", "dart", "love",
      "because", "i", "the", "programming", "in", "dart",
      "and", "i", "and", "love", "on"
    ];

    for (int i = 0; i < generatedText.length; i++) {
      if (generatedText[i] != expectedSequence[i]) {
        throw Exception("❌ TEST FAILED: Output diverged at index $i. Expected '${expectedSequence[i]}', got '${generatedText[i]}'.");
      }
    }

    print("\n🎉 E2E TEST PASSED PERFECTLY!");
    print("📝 Output: ${generatedText.join(" ")}");

  } catch (e) {
    print("\n🚨 TEST CRASHED: $e");
    // Rethrow to ensure the CLI exits with an error code (useful for CI/CD pipelines)
    rethrow;

  } finally {
    // 6. Tear Down: Guaranteed Memory Cleanup
    print("🧹 Cleaning up GPU memory...");
    sequence?.free();
    probTable?.free();
    historyBatch?.free();
    nextProbs?.free();
    CudaEngine.dispose();
  }
}