

/*Future<void> main() async {
  // 1. Dataset Setup
  Map<String, int> vocabulary = {
    '<pad>': 0, 'i': 1, 'love': 2, 'this': 3, 'movie': 4, 'is': 5, 'great': 6,
    'a': 7, 'bad': 8, 'film': 9, 'hate': 10, 'terrible': 11, 'good': 12,
    'feel': 13, 'it': 14,
  };
  int vocabSize = vocabulary.length;

  List<List<dynamic>> rawData = [
    ['i love this movie', 1.0], ['this film is great', 1.0],
    ['i hate this film', 0.0], ['this is a terrible movie', 0.0],
    ['this movie is good', 1.0], ['i feel great', 1.0],
    ['this is bad', 0.0], ['i hate it', 0.0],
  ];

  List<Vector> inputs = [];
  List<Vector> targets = [];

  for (int i = 0; i < rawData.length; i = i + 1) {
    String sentence = rawData[i][0] as String;
    double label = rawData[i][1] as double;

    List<String> words = sentence.split(' ');
    Vector tokenized = [];
    for (int j = 0; j < words.length; j = j + 1) {
      if (vocabulary.containsKey(words[j])) {
        tokenized.add(vocabulary[words[j]]!.toDouble());
      }
    }
    inputs.add(tokenized);
    targets.add([label]);
  }

  // 2. Model Definition
  int dModel = 16;
  int numHeads = 2;
  int dff = 32;
  int maxSequenceLength = 10;

  SNetwork sentimentClassifier = SNetwork([
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(maxSequenceLength, dModel),
    TransformerEncoderBlock(dModel, numHeads, dff),
    TransformerEncoderBlock(dModel, numHeads, dff),
    GlobalAveragePooling1D(),
    DenseLayer(1, activation: Sigmoid()),
  ], name: 'Stress-Test-Transformer');

  // Initialize
  Tensor<Vector> dummyInput = Tensor<Vector>([1.0, 2.0, 3.0]);
  sentimentClassifier.predict(dummyInput);

  Adam optimizer = Adam(sentimentClassifier.parameters, learningRate: 0.001);
  sentimentClassifier.compile(configuredOptimizer: optimizer);

  // 3. Stress Test Training Loop
  int totalEpochs = 100000;
  int logInterval = 100;
  Stopwatch intervalWatch = Stopwatch();

  print('Starting Stress Test: $totalEpochs epochs...');
  print('Monitoring timing every $logInterval epochs.\n');

  for (int epoch = 0; epoch < totalEpochs; epoch = epoch + 1) {
    if (epoch % logInterval == 0) {
      intervalWatch.reset();
      intervalWatch.start();
    }

    double epochLoss = 0.0;

    for (int i = 0; i < inputs.length; i = i + 1) {
      // Create fresh tensors for this step
      Tensor<Vector> inputTensor = Tensor<Vector>(inputs[i]);
      Tensor<Vector> targetTensor = Tensor<Vector>(targets[i]);

      // Reset gradients
      for (int p = 0; p < sentimentClassifier.parameters.length; p = p + 1) {
        sentimentClassifier.parameters[p].zeroGrad();
      }

      // Forward
      Tensor<Vector> output = sentimentClassifier.forward(inputTensor) as Tensor<Vector>;

      // Loss
      Tensor<Scalar> loss = mse(output, targetTensor);
      epochLoss = epochLoss + loss.data[0];

      // Backward and Update
      loss.backward();
      loss.printGraph();
      optimizer.step();

      // IMPORTANT: If execution time climbs, check if these are freeing correctly
      // or if intermediate tensors inside layers are hanging in memory.
      inputTensor.free();
      targetTensor.free();
      // Note: Intermediate tensors in the graph are still allocated!
    }

    if ((epoch + 1) % logInterval == 0) {
      intervalWatch.stop();
      double avgTime = intervalWatch.elapsedMilliseconds / logInterval;
      double avgLoss = epochLoss / inputs.length;

      print('Epoch ${epoch + 1} | Avg Loss: ${avgLoss.toStringAsFixed(6)} | Timing: ${avgTime.toStringAsFixed(2)} ms/epoch');

      // Optional: Force a small GC hint if running in a standard Dart VM
      // but ffi pointers won't be affected by GC.
    }
  }

  print('\nStress Test Complete.');
}*/