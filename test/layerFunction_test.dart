

import 'dart:math';

import 'package:flutter_ml_web_gpu/full_library.dart';
import 'package:flutter_ml_web_gpu/gpu_version/SeqModel.dart';
import 'package:flutter_ml_web_gpu/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml_web_gpu/gpu_version/optimizer/adam.dart';
import 'package:flutter_ml_web_gpu/logger.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  setUpAll(() async {
    await GPUEngine.initialize(target: Target.cuda);
  });

  test('Training and Static Prediction', () {
    GPUTensor<Matrix> trainInput = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 1.0],
      <double>[2.0, 2.0]
    ]);

    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[2.0, 2.0],
      <double>[4.0, 4.0]
    ]);

    SeqModel<Matrix, Matrix> model = SeqModel<Matrix, Matrix>(
      <TapeLayer>[DenseTL(2)],
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params) {
        return SGDGPU(params, 0.01);
      },
    );

    model.compile();

    int epochs = 1000;
    for (int i = 0; i < epochs; i = i + 1) {
      model.runTraining();
    }

    model.loss?.toCpu();
    Logger.green('Final Training Loss: ${model.loss?.value}');

    GPUTensor<Matrix> inferInput = GPUTensor<Matrix>(<List<double>>[
      <double>[3.0, 3.0]
    ]);

    GPUTensor<Matrix> inferResult = model.predict(inferInput);

    model.runForward();
    inferResult.toCpu();
    Logger.blue('Prediction for [3.0, 3.0]: ${inferResult.value}');

    List<double> newInferData = <double>[4.0, 4.0];
    model.runForward(inputData: newInferData);
    inferResult.toCpu();
    Logger.blue('Prediction for [4.0, 4.0]: ${inferResult.value}');

    model.free();
  });
  test('GPU Engine Benchmark - Deep Network Static Unrolling', () {
    int batchSize = 1024;
    int inputFeatures = 64 * 64;
    int outputFeatures = 16;

    Random rnd = Random(42);

    List<List<double>> rawInput = <List<double>>[];
    List<List<double>> rawTarget = <List<double>>[];

    for (int i = 0; i < batchSize; i = i + 1) {
      List<double> inRow = <double>[];
      for (int j = 0; j < inputFeatures; j = j + 1) {
        inRow.add(rnd.nextDouble());
      }
      rawInput.add(inRow);

      List<double> outRow = <double>[];
      for (int j = 0; j < outputFeatures; j = j + 1) {
        outRow.add(rnd.nextDouble());
      }
      rawTarget.add(outRow);
    }

    GPUTensor<Matrix> trainInput = GPUTensor<Matrix>(rawInput);
    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(rawTarget);

    SeqModel<Matrix, Matrix> model = SeqModel<Matrix, Matrix>(
      <TapeLayer>[
        DenseTL(128),
        GeluLayerMatrixTL(),
        DenseTL(64),
        GeluLayerMatrixTL(),
        DenseTL(outputFeatures)
      ],
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params) {
        return AdamGPU(params, 0.001);
      },
    );

    Stopwatch timer = Stopwatch();

    Logger.blue('Compiling static tapes...');
    timer.start();
    model.compile();
    timer.stop();
    Logger.green('Compilation took: ${timer.elapsedMilliseconds} ms');

    Logger.blue('Starting high-speed training loop (20 seconds max) for Batch Size $batchSize...');
    timer.reset();
    timer.start();

    int actualEpochs = 0;

    // Loop scales up dynamically until 20 seconds have passed
    while (timer.elapsedMilliseconds < 20000) {
      model.runTraining();
      actualEpochs = actualEpochs + 1;
    }

    timer.stop();
    model.loss?.toCpu();

    double seconds = timer.elapsedMilliseconds / 1000.0;
    double epochsPerSec = actualEpochs / seconds;

    // Approximate theoretical FLOPs strictly based on dense matrix multiplications (M * N * 2K)
    double dense1Flops = batchSize * 128 * inputFeatures * 2.0;
    double dense2Flops = batchSize * 64 * 128 * 2.0;
    double dense3Flops = batchSize * outputFeatures * 64 * 2.0;

    double forwardFlops = dense1Flops + dense2Flops + dense3Flops;

    // Backpropagation scales to roughly 2x the forward pass (gradients for weights and inputs)
    double totalFlopsPerEpoch = forwardFlops * 3.0;

    double totalFlopsAchieved = totalFlopsPerEpoch * actualEpochs;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green('Final Training Loss: ${model.loss?.value}');
    Logger.green('Completed $actualEpochs epochs in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green('Engine Performance: ${epochsPerSec.toStringAsFixed(2)} Epochs/sec');
    Logger.green('Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    model.free();
  });
  test('GPU Engine Benchmark - Pure MatMul Isolated', () {
    // 2048 x 2048 x 2048 = ~17.17 GFLOPs per execution
    int m = 2048;
    int k = 2048;
    int n = 2048;

    Random rnd = Random(42);

    List<List<double>> rawA = <List<double>>[];
    for (int i = 0; i < m; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < k; j = j + 1) {
        row.add(rnd.nextDouble());
      }
      rawA.add(row);
    }

    List<List<double>> rawB = <List<double>>[];
    for (int i = 0; i < k; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < n; j = j + 1) {
        row.add(rnd.nextDouble());
      }
      rawB.add(row);
    }

    GPUTensor<Matrix> tensorA = GPUTensor<Matrix>(rawA);
    GPUTensor<Matrix> tensorB = GPUTensor<Matrix>(rawB);
    GPUTensor<Matrix> tensorC = GPUTensor<Matrix>.empty(<int>[m, n]);

    CommandBuffer tape = CommandBuffer();
    matMulGPU(tensorA, tensorB, tape, outTensor: tensorC);

    Stopwatch timer = Stopwatch();
    Logger.blue('Starting high-speed pure MatMul loop (20 seconds max)...');
    timer.start();

    int actualSteps = 0;

    while (timer.elapsedMilliseconds < 20000) {
      GPUEngine.run(tape.bytes());
      actualSteps = actualSteps + 1;
    }

    timer.stop();
    double seconds = timer.elapsedMilliseconds / 1000.0;

    // Standard FLOP calculation for Matrix Multiplication: 2 * M * K * N
    double flopsPerStep = 2.0 * m * k * n;
    double totalFlopsAchieved = flopsPerStep * actualSteps;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green('Completed $actualSteps MatMul steps in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green('Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    tensorA.free();
    tensorB.free();
    tensorC.free();
  });
  test('GPU Engine Benchmark - Absolute VRAM Saturation MatMul', () {
    // 24576 x 24576 = ~603.9 million elements per matrix.
    // At 4 bytes per float, this equals ~2.41 GB per matrix.
    // 3 Matrices (A, B, C) = ~7.23 GB of VRAM allocated instantly.
    // Compute cost: ~29.67 TFLOPs per single execution step.
    int m = 2048;//24576;
    int k = 2048;//24576;
    int n = 2048;//24576;

    Logger.blue('Allocating ~7.23 GB directly in VRAM...');

    // Bypassing CPU initialization to prevent system RAM Out-Of-Memory crashes
    GPUTensor<Matrix> tensorA = GPUTensor<Matrix>.empty(<int>[m, k]);
    GPUTensor<Matrix> tensorB = GPUTensor<Matrix>.empty(<int>[k, n]);
    GPUTensor<Matrix> tensorC = GPUTensor<Matrix>.empty(<int>[m, n]);

    CommandBuffer tape = CommandBuffer();
    matMulGPU(tensorA, tensorB, tape, outTensor: tensorC);

    // We only run 5 steps because each step is a massive 29.67 TFLOP workload
    int steps = 5;

    Logger.blue('Starting VRAM saturation test for $steps steps...');
    Stopwatch timer = Stopwatch();
    timer.start();

    for (int i = 0; i < steps; i = i + 1) {
      GPUEngine.run(tape.bytes());
    }

    timer.stop();
    double seconds = timer.elapsedMilliseconds / 1000.0;

    // Standard FLOP calculation for Matrix Multiplication: 2 * M * K * N
    double flopsPerStep = 2.0 * m * k * n;
    double totalFlopsAchieved = flopsPerStep * steps;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green('Completed $steps massive MatMul steps in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green('Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    tensorA.free();
    tensorB.free();
    tensorC.free();
  });
  test('Minimalist Transformer Sequence-to-Sequence', () {
    // 1. Setup Architecture Dimensions
    int vocabSize = 10;
    int dModel = 16;
    int numHeads = 4;
    int dff = 32;
    int maxSeqLength = 8;

    // 2. Training Data: A sequence of 4 token IDs
    GPUTensor<Vector> trainInput = GPUTensor<Vector>(<double>[1.0, 3.0, 5.0, 2.0]);

    // Training Target: One-hot encoded matrix for the target outputs (Shape: 4 x 10)
    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 1.0 -> 2.0
      <double>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 3.0 -> 4.0
      <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // Target for input 5.0 -> 6.0
      <double>[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 2.0 -> 3.0
    ]);

    // 3. Define the Transformer Pipeline
    List<TapeLayer> layers = <TapeLayer>[
      // Embeds the 1D token IDs into a 2D Matrix of shape [SeqLength, dModel]
      EmbeddingTL(vocabSize, dModel),

      // Injects sine/cosine spatial frequencies
      PositionalEncodingTL(maxSeqLength, dModel),

      // The heavy-lifter: Multi-Head Self Attention + Residuals + LayerNorm + FFN
      TransformerEncoderBlockTapeLayer(dModel, numHeads, dff),

      // Projects the 16-dimensional features back into the 10-dimensional vocabulary space
      DenseTL(vocabSize)
    ];

    // 4. Initialize the Sequential Wrapper
    SeqModel<Vector, Matrix> model = SeqModel<Vector, Matrix>(
      layers,
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params) {
        return AdamGPU(params, 0.01);
      },
    );

    Logger.blue('Compiling Transformer Tapes...');
    model.compile();
    Logger.green('Compilation Successful.');

    // 5. Run the Training Loop
    int epochs = 250;
    Logger.blue('Starting Training for $epochs epochs...');

    for (int i = 0; i < epochs; i = i + 1) {
      model.runTraining();
    }

    model.loss?.toCpu();
    Logger.green('Final Training Loss: ${model.loss?.value}');

    // 6. Test Static Inference with dynamic sequence lengths!
    // Notice how the inference sequence length (2) is different from training (4).
    // The model.predict() gracefully handles reshaping the internal caches.
    GPUTensor<Vector> inferInput = GPUTensor<Vector>(<double>[1.0, 3.0]);
    GPUTensor<Matrix> inferResult = model.predict(inferInput);

    model.runForward();
    inferResult.toCpu();

    Logger.blue('--- Inference Results ---');
    Logger.blue('Input Tokens: [1.0, 3.0]');

    // Prints the output probabilities/logits for the 2 tokens across the 10 vocab classes
    List<List<double>> outputMatrix = inferResult.value;
    for (int i = 0; i < outputMatrix.length; i = i + 1) {
      Logger.blue('Token $i Output Logits: ${outputMatrix[i]}');
    }

    // Cleanup VRAM
    model.free();
  });

}
