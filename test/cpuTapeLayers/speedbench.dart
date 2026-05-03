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

void runBenchmark<I, O>(String name, Layer<I, O> layer, Tensor<I> input, int iterations) {
  print('--- Benchmarking: $name ---');

  layer.build(input);

  for (int i = 0; i < 5; i = i + 1) {
    Tensor<O> out = layer.forward(input);
    Tensor<Scalar> loss = pseudoLoss(out);
    loss.backward();
  }

  Stopwatch fwWatch = Stopwatch();
  Stopwatch bwWatch = Stopwatch();

  for (int i = 0; i < iterations; i = i + 1) {
    for (int p = 0; p < layer.parameters.length; p = p + 1) {
      layer.parameters[p].zeroGrad();
    }
    input.zeroGrad();

    fwWatch.start();
    Tensor<O> out = layer.forward(input);
    fwWatch.stop();

    Tensor<Scalar> loss = pseudoLoss(out);

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

void main() {
  print('========================================');
  print('   LAYER SPEED BENCHMARK (CPU ONLY)     ');
  print('========================================\n');

  int iterations = 100;

  // 1. MaxPooling2D
  MaxPooling2DLayer maxPool = MaxPooling2DLayer(poolSize: 2, stride: 2);
  Tensor<Matrix> poolInput = generateMatrix(128, 128);
  runBenchmark<Matrix, Matrix>('MaxPooling2D (128x128)', maxPool, poolInput, iterations);

  // 2. ReLULayerMatrix
  ReLULayerMatrix reluMatrix = ReLULayerMatrix();
  Tensor<Matrix> reluInput = generateMatrix(128, 128);
  runBenchmark<Matrix, Matrix>('ReLU Matrix (128x128)', reluMatrix, reluInput, iterations);

  // 3. MultiLSTMLayer (Assuming 3 stacked LSTMs)
  // Input: Seq=60, features=32 -> Hidden=64
  MultiTierLSTMLayer multiLstm = MultiTierLSTMLayer(64, tierClockCycles: [6,3]);
  Tensor<Matrix> multiLstmInput = generateMatrix(60, 32);
  runBenchmark<Matrix, Vector>('MultiLSTM (Seq:60, 3 layers, 64 hidden)', multiLstm, multiLstmInput, iterations);

  print('========================================');
  print('         BENCHMARK COMPLETE             ');
  print('========================================');
}