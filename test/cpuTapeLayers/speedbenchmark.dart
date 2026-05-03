import 'dart:math';

import 'package:flutter_ml/full_library.dart';

Tensor<Vector> generateVector(int size) {
  Random random = Random();
  Vector v = [];
  for (int i = 0; i < size; i = i + 1) {
    v.add(random.nextDouble());
  }
  return Tensor<Vector>(v);
}
Tensor<Matrix> generateMatrix(int rows, int cols) {
  Random random = Random();
  Matrix m = [];
  for (int i = 0; i < rows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < cols; j = j + 1) {
      row.add(random.nextDouble());
    }
    m.add(row);
  }
  return Tensor<Matrix>(m);
}
Tensor<Tensor3D> generateTensor3D(int depth, int rows, int cols) {
  Random random = Random();
  Tensor3D t3d = [];
  for (int d = 0; d < depth; d = d + 1) {
    Matrix m = [];
    for (int i = 0; i < rows; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < cols; j = j + 1) {
        row.add(random.nextDouble());
      }
      m.add(row);
    }
    t3d.add(m);
  }
  return Tensor<Tensor3D>(t3d);
}

// ─────────────────────────────────────────────────────── //
// UNIVERSAL LOSS FOR BACKWARD BENCHMARKING
// ─────────────────────────────────────────────────────── //

Tensor<Scalar> pseudoLoss<T>(Tensor<T> out) {
  double sum = 0.0;
  for (int i = 0; i < out.data.length; i = i + 1) {
    sum = sum + out.data[i];
  }

  Tensor<Scalar> loss = Tensor<Scalar>(sum);
  loss.creator = Node(
      [out],
          () {
        for (int i = 0; i < out.data.length; i = i + 1) {
          out.grad[i] = out.grad[i] + loss.grad[0];
        }
      },
      opName: 'pseudo_sum_loss',
      cost: out.data.length
  );

  return loss;
}

// ─────────────────────────────────────────────────────── //
// BENCHMARK RUNNER
// ─────────────────────────────────────────────────────── //

void runBenchmark<I, O>(String name, Layer<I, O> layer, Tensor<I> input, int iterations, {bool showGraph = false}) {
  print('--- Benchmarking: $name ---');

  // 1. Build the layer
  layer.build(input);

  // 2. Warmup (to trigger JIT compilation and memory allocation)
  for (int i = 0; i < 5; i = i + 1) {
    Tensor<O> out = layer.forward(input);
    Tensor<Scalar> loss = pseudoLoss(out);

    // Print the compute graph on the very first warmup pass if requested
    if (showGraph && i == 0) {
      loss.printGraph();
      print('');
    }

    loss.backward();
  }

  Stopwatch fwWatch = Stopwatch();
  Stopwatch bwWatch = Stopwatch();

  // 3. Main Benchmark Loop
  for (int i = 0; i < iterations; i = i + 1) {
    // Zero gradients before each step
    for (int p = 0; p < layer.parameters.length; p = p + 1) {
      layer.parameters[p].zeroGrad();
    }
    input.zeroGrad();

    // Time Forward Pass
    fwWatch.start();
    Tensor<O> out = layer.forward(input);
    fwWatch.stop();

    Tensor<Scalar> loss = pseudoLoss(out);

    // Time Backward Pass
    bwWatch.start();
    loss.backward();
    bwWatch.stop();
  }

  double avgFw = fwWatch.elapsedMilliseconds / iterations;
  double avgBw = bwWatch.elapsedMilliseconds / iterations;

  print('Forward Pass:  ${avgFw.toStringAsFixed(2)} ms / step');
  print('Backward Pass: ${avgBw.toStringAsFixed(2)} ms / step');
  print('Total Time:    ${(avgFw + avgBw).toStringAsFixed(2)} ms / step\n');
}

// ─────────────────────────────────────────────────────── //
// MAIN EXECUTABLE
// ─────────────────────────────────────────────────────── //

void main() {
  print('========================================');
  print('   LAYER SPEED BENCHMARK (CPU ONLY)     ');
  print('========================================\n');

  int iterations = 50;

  // 1. AveragePooling2DLayer
  AveragePooling2DLayer avgPool = AveragePooling2DLayer(poolSize: 2, stride: 2);
  Tensor<Matrix> poolInput = generateMatrix(128, 128);
  runBenchmark<Matrix, Matrix>('AveragePooling2D (128x128)', avgPool, poolInput, iterations);

  // 2. BatchNorm1D
  BatchNorm1D bn1d = BatchNorm1D(1024);
  Tensor<Vector> bn1dInput = generateVector(1024);
  runBenchmark<Vector, Vector>('BatchNorm1D (1024 features)', bn1d, bn1dInput, iterations);

  // 3. BatchNorm2D
  BatchNorm2D bn2d = BatchNorm2D(16);
  Tensor<Tensor3D> bn2dInput = generateTensor3D(16, 64, 64);
  runBenchmark<Tensor3D, Tensor3D>('BatchNorm2D (16 channels, 64x64)', bn2d, bn2dInput, iterations);

  // 4. Conv2DLayer
  Conv2DLayer conv2d = Conv2DLayer(16, 3, padding: 'same');
  Tensor<Matrix> convInput = generateMatrix(64, 64);
  // Setting showGraph: true here to visualize the convolution bottleneck
  runBenchmark<Matrix, Tensor3D>('Conv2D (16 filters, 3x3, 64x64 in)', conv2d, convInput, iterations, showGraph: true);

  // 5. ConvLSTMLayer
  ConvLSTMLayer convLstm = ConvLSTMLayer(16, 3);
  Tensor<Tensor3D> convLstmInput = generateTensor3D(10, 32, 32);
  runBenchmark<Tensor3D, Matrix>('ConvLSTM (Seq:10, 16 filters, 3x3, 32x32 in)', convLstm, convLstmInput, iterations);

  print('========================================');
  print('         BENCHMARK COMPLETE             ');
  print('========================================');
}