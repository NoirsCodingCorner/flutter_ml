import 'dart:math';

import 'package:flutter_ml/full_library.dart';
import 'package:flutter_ml/gpu_version/seq_model.dart';
import 'package:flutter_ml/gpu_version/ffi/gpu_engine.dart';
import 'package:flutter_ml/gpu_version/optimizer/adam.dart';
import 'package:flutter_ml/logger.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  setUpAll(() async {
    await GPUEngine.initialize(target: Target.auto);
  });

  tearDownAll(() {
    GPUEngine.dispose();
  });

  test('Training and Static Prediction', () {
    AllocBuffer dataAlloc = AllocBuffer();

    GPUTensor<Matrix> trainInput = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 1.0],
      <double>[2.0, 2.0]
    ], dataAlloc);

    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[2.0, 2.0],
      <double>[4.0, 4.0]
    ], dataAlloc);

    dataAlloc.execute();

    SeqModel<Matrix, Matrix> model = SeqModel<Matrix, Matrix>(
      <TapeLayer>[DenseTL(2)],
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params, AllocBuffer allocTape) {
        return SGDGPU(params, 0.01);
      },
    );

    model.compile();

    int epochs = 1000;
    for (int i = 0; i < epochs; i = i + 1) {
      model.runTraining();
    }

    model.loss?.dataToCpu();
    Logger.green('Final Training Loss: ${model.loss?.value}');

    AllocBuffer inferAlloc = AllocBuffer();
    GPUTensor<Matrix> inferInput = GPUTensor<Matrix>(<List<double>>[
      <double>[3.0, 3.0]
    ], inferAlloc);
    inferAlloc.execute();

    GPUTensor<Matrix> inferResult = model.predict(inferInput);

    model.runForward();
    inferResult.dataToCpu();
    Logger.blue('Prediction for [3.0, 3.0]: ${inferResult.value}');

    List<double> newInferData = <double>[4.0, 4.0];
    model.runForward(inputData: newInferData);
    inferResult.dataToCpu();
    Logger.blue('Prediction for [4.0, 4.0]: ${inferResult.value}');

    model.free();
    inferAlloc.freeAll();
    dataAlloc.freeAll();
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

    AllocBuffer dataAlloc = AllocBuffer();
    GPUTensor<Matrix> trainInput = GPUTensor<Matrix>(rawInput, dataAlloc);
    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(rawTarget, dataAlloc);
    dataAlloc.execute();

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
      optimizerBuilder: (List<GPUTensor> params, AllocBuffer allocTape) {
        return AdamGPU(params, 0.001, allocTape);
      },
    );

    Stopwatch timer = Stopwatch();

    Logger.blue('Compiling static tapes...');
    timer.start();
    model.compile();
    timer.stop();
    Logger.green('Compilation took: ${timer.elapsedMilliseconds} ms');

    Logger.blue(
        'Starting high-speed training loop (20 seconds max) for Batch Size $batchSize...');
    timer.reset();
    timer.start();

    int actualEpochs = 0;

    while (timer.elapsedMilliseconds < 20000) {
      model.runTraining();
      actualEpochs = actualEpochs + 1;
    }

    timer.stop();
    model.loss?.dataToCpu();

    double seconds = timer.elapsedMilliseconds / 1000.0;
    double epochsPerSec = actualEpochs / seconds;

    double dense1Flops = batchSize * 128 * inputFeatures * 2.0;
    double dense2Flops = batchSize * 64 * 128 * 2.0;
    double dense3Flops = batchSize * outputFeatures * 64 * 2.0;

    double forwardFlops = dense1Flops + dense2Flops + dense3Flops;
    double totalFlopsPerEpoch = forwardFlops * 3.0;

    double totalFlopsAchieved = totalFlopsPerEpoch * actualEpochs;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green('Final Training Loss: ${model.loss?.value}');
    Logger.green(
        'Completed $actualEpochs epochs in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green(
        'Engine Performance: ${epochsPerSec.toStringAsFixed(2)} Epochs/sec');
    Logger.green(
        'Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    model.free();
    dataAlloc.freeAll();
  });

  test('GPU Engine Benchmark - Pure MatMul Isolated', () {
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

    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> tensorA = GPUTensor<Matrix>(rawA, allocTape);
    GPUTensor<Matrix> tensorB = GPUTensor<Matrix>(rawB, allocTape);

    CommandBuffer tape = CommandBuffer();
    matMulGPU(tensorA, tensorB, tape, allocTape);
    allocTape.execute();

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

    double flopsPerStep = 2.0 * m * k * n;
    double totalFlopsAchieved = flopsPerStep * actualSteps;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green(
        'Completed $actualSteps MatMul steps in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green(
        'Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    allocTape.freeAll();
  });

  test('GPU Engine Benchmark - Absolute VRAM Saturation MatMul', () {
    int m = 2048; //24576;
    int k = 2048; //24576;
    int n = 2048; //24576;

    Logger.blue('Allocating ~7.23 GB directly in VRAM...');

    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> tensorA = GPUTensor<Matrix>.empty(<int>[m, k], allocTape);
    GPUTensor<Matrix> tensorB = GPUTensor<Matrix>.empty(<int>[k, n], allocTape);

    CommandBuffer tape = CommandBuffer();
    matMulGPU(tensorA, tensorB, tape, allocTape);
    allocTape.execute();

    int steps = 5;

    Logger.blue('Starting VRAM saturation test for $steps steps...');
    Stopwatch timer = Stopwatch();
    timer.start();

    for (int i = 0; i < steps; i = i + 1) {
      GPUEngine.run(tape.bytes());
    }

    timer.stop();
    double seconds = timer.elapsedMilliseconds / 1000.0;

    double flopsPerStep = 2.0 * m * k * n;
    double totalFlopsAchieved = flopsPerStep * steps;
    double flopsPerSecond = totalFlopsAchieved / seconds;

    double gflops = flopsPerSecond / 1000000000.0;
    double tflops = gflops / 1000.0;

    Logger.green(
        'Completed $steps massive MatMul steps in ${seconds.toStringAsFixed(2)} seconds');
    Logger.green(
        'Compute Performance: ${gflops.toStringAsFixed(2)} GFLOPS (${tflops.toStringAsFixed(4)} TFLOPS)');

    allocTape.freeAll();
  });

  test('Minimalist Transformer Sequence-to-Sequence', () {
    int vocabSize = 10;
    int dModel = 16;
    int numHeads = 4;
    int dff = 32;
    int maxSeqLength = 8;

    AllocBuffer dataAlloc = AllocBuffer();

    GPUTensor<Vector> trainInput =
        GPUTensor<Vector>(<double>[1.0, 3.0, 5.0, 2.0], dataAlloc);

    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dataAlloc);

    dataAlloc.execute();

    List<TapeLayer> layers = <TapeLayer>[
      EmbeddingTL(vocabSize, dModel),
      PositionalEncodingTL(maxSeqLength, dModel),
      TransformerEncoderBlockTapeLayer(dModel, numHeads, dff),
      DenseTL(vocabSize)
    ];

    SeqModel<Vector, Matrix> model = SeqModel<Vector, Matrix>(
      layers,
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params, AllocBuffer allocTape) {
        return AdamGPU(params, 0.01, allocTape);
      },
    );

    Logger.blue('Compiling Transformer Tapes...');
    model.compile();
    Logger.green('Compilation Successful.');

    int epochs = 250;
    Logger.blue('Starting Training for $epochs epochs...');

    for (int i = 0; i < epochs; i = i + 1) {
      model.runTraining();
    }

    model.loss?.dataToCpu();
    Logger.green('Final Training Loss: ${model.loss?.value}');

    AllocBuffer inferAlloc = AllocBuffer();
    GPUTensor<Vector> inferInput =
        GPUTensor<Vector>(<double>[1.0, 3.0], inferAlloc);
    inferAlloc.execute();

    GPUTensor<Matrix> inferResult = model.predict(inferInput);

    model.runForward();
    inferResult.dataToCpu();

    Logger.blue('--- Inference Results ---');
    Logger.blue('Input Tokens: [1.0, 3.0]');

    List<List<double>> outputMatrix = inferResult.value;
    for (int i = 0; i < outputMatrix.length; i = i + 1) {
      Logger.blue('Token $i Output Logits: ${outputMatrix[i]}');
    }

    model.free();
    inferAlloc.freeAll();
    dataAlloc.freeAll();
  });
}
