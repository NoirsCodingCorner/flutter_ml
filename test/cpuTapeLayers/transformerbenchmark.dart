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

Tensor<Vector> generateIndexVector(int size, int maxIndex) {
  Random random = Random();
  Vector v = [];
  for (int i = 0; i < size; i = i + 1) {
    v.add((random.nextDouble() * (maxIndex - 1)).roundToDouble());
  }
  return Tensor<Vector>(v);
}

Tensor<Matrix> generateIndexMatrix(int rows, int cols, int maxIndex) {
  Random random = Random();
  Matrix m = [];
  for (int i = 0; i < rows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < cols; j = j + 1) {
      row.add((random.nextDouble() * (maxIndex - 1)).roundToDouble());
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
  print(' TRANSFORMER LAYER BENCHMARK (CPU ONLY) ');
  print('========================================\n');

  int iterations = 100;

  // Standard dimension variables for benchmarking
  int vocabSize = 10000;
  int embedDim = 128;
  int seqLength = 64;
  int batchSize = 16;
  int dModel = 128;
  int numHeads = 8;
  int dff = 512;

  // 1. EmbeddingLayer (1D input)
  EmbeddingLayer embLayer = EmbeddingLayer(vocabSize, embedDim);
  Tensor<Vector> embInput = generateIndexVector(seqLength, vocabSize);
  runBenchmark<Vector, Matrix>('EmbeddingLayer (Seq: $seqLength, Dim: $embedDim)', embLayer, embInput, iterations);

  // 2. EmbeddingLayerMatrix (2D Input - Batch processing)
  EmbeddingLayerMatrix embLayerMatrix = EmbeddingLayerMatrix(vocabSize, embedDim);
  Tensor<Matrix> embMatrixInput = generateIndexMatrix(batchSize, seqLength, vocabSize);
  runBenchmark<Matrix, Tensor3D>('EmbeddingLayerMatrix (Batch: $batchSize, Seq: $seqLength, Dim: $embedDim)', embLayerMatrix, embMatrixInput, iterations);

  // 3. GlobalAveragePooling1D
  GlobalAveragePooling1D globalPool = GlobalAveragePooling1D();
  Tensor<Matrix> poolInput = generateMatrix(seqLength, dModel);
  runBenchmark<Matrix, Vector>('GlobalAveragePooling1D (Seq: $seqLength, Feat: $dModel)', globalPool, poolInput, iterations);

  // 4. LayerNormalization
  LayerNormalization layerNorm = LayerNormalization();
  Tensor<Matrix> normInput = generateMatrix(seqLength, dModel);
  runBenchmark<Matrix, Matrix>('LayerNormalization (Seq: $seqLength, Feat: $dModel)', layerNorm, normInput, iterations);

  // 5. PositionalEncoding
  PositionalEncoding posEncoding = PositionalEncoding(1024, dModel);
  Tensor<Matrix> posInput = generateMatrix(seqLength, dModel);
  runBenchmark<Matrix, Matrix>('PositionalEncoding (Seq: $seqLength, dModel: $dModel)', posEncoding, posInput, iterations);

  // 6. MultiHeadAttention
  MultiHeadAttention mha = MultiHeadAttention(dModel, numHeads);
  Tensor<Matrix> mhaInput = generateMatrix(seqLength, dModel);
  runBenchmark<Matrix, Matrix>('MultiHeadAttention (Seq: $seqLength, dModel: $dModel, Heads: $numHeads)', mha, mhaInput, iterations);

  // 7. TransformerEncoderBlock
  TransformerEncoderBlock transformerBlock = TransformerEncoderBlock(dModel, numHeads, dff);
  Tensor<Matrix> blockInput = generateMatrix(seqLength, dModel);
  runBenchmark<Matrix, Matrix>('TransformerEncoderBlock (Seq: $seqLength, dModel: $dModel, dff: $dff)', transformerBlock, blockInput, iterations);

  print('========================================');
  print('         BENCHMARK COMPLETE             ');
  print('========================================');
}