import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/full_library.dart';

void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  double learningRate = 0.01;
  int epochsPerTest = 100;

  List<int> sequences = <int>[1,2, 4, 8, 16, 32, 64, 128, 512, 2048];
  List<int> dimensions = <int>[256, 1024, 4096,8192,16384];

  print('====================================================================================================');
  print('                  GPU LAYER NORM & POSITIONAL ENCODING THROUGHPUT BENCHMARK                         ');
  print('====================================================================================================');

  // ===================================================================================================
  // BENCHMARK 1: LAYER NORMALIZATION
  // ===================================================================================================
  print('\n--- 1. LAYER NORMALIZATION BENCHMARK ---');
  print(' SeqLen | dModel | Avg Time(ms) | Throughput (GB/s) | Final Loss');
  print('----------------------------------------------------------------------------------------------------');

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int dModel = dimensions[d];

      List<List<double>> hInput = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add((random.nextDouble() * 2.0) - 1.0);
        }
        hInput.add(row);
      }

      List<List<double>> hTarget = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add(0.5);
        }
        hTarget.add(row);
      }

      GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);
      GPUTensor<Matrix> target = GPUTensor<Matrix>(hTarget);

      LayerNormalizationTL normLayer = LayerNormalizationTL(dModel,epsilon: 1e-5);
      normLayer.build(input);

      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> output = normLayer.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
      GPUTensor<Scalar> loss = GPUTensor<Scalar>(0.0);

      fTape.putInt(OP_MSE_LOSS_FORWARD);
      fTape.putString(output.id);
      fTape.putString(target.id);
      fTape.putString(loss.id);

      loss.creator = GPUNode(
        <GPUTensor>[output, target],
            (CommandBuffer bTape) {
          bTape.putInt(OP_MSE_LOSS_BACKWARD);
          bTape.putString('${output.id}_grad');
          bTape.putString(output.id);
          bTape.putString(target.id);
          bTape.putString('${loss.id}_grad');
        },
        opName: 'mse_loss_manual',
      );

      Uint8List forwardBytes = fTape.bytes();

      CommandBuffer bTape = CommandBuffer();
      SGDGPU optimizer = SGDGPU(normLayer.parameters, learningRate);

      optimizer.zeroGrad(bTape);
      for (int i = 0; i < intermediates.length; i = i + 1) {
        bTape.putInt(OP_ZERO_GRAD);
        bTape.putString('${intermediates[i].id}_grad');
      }
      bTape.putInt(OP_ZERO_GRAD);
      bTape.putString('${input.id}_grad');
      bTape.putInt(OP_ZERO_GRAD);
      bTape.putString('${output.id}_grad');

      bTape.putInt(OP_FILL);
      bTape.putString('${loss.id}_grad');
      bTape.putFloat(1.0);

      loss.backward(bTape);
      Uint8List backwardBytes = bTape.bytes();

      CommandBuffer oTape = CommandBuffer();
      optimizer.step(oTape);
      Uint8List optimizeBytes = oTape.bytes();

      CudaEngine.run(forwardBytes);

      Stopwatch runSw = Stopwatch();
      runSw.start();
      for (int epoch = 1; epoch <= epochsPerTest; epoch = epoch + 1) {
        CudaEngine.run(forwardBytes);
        CudaEngine.run(backwardBytes);
        CudaEngine.run(optimizeBytes);
      }
      runSw.stop();

      loss.toCpu();
      double currentLoss = loss.value;

      double avgEpochSec = (runSw.elapsedMicroseconds / 1000000.0) / epochsPerTest;
      double avgEpochMs = avgEpochSec * 1000.0;

      // Rough Memory Traffic for Norm:
      // Fwd: Read Input, Write Out. Bwd: Read GradOut, Read Input, Write GradIn.
      // Total approx 5 * (seqLength * dModel * 4 bytes)
      double totalBytesStep = 5.0 * seqLength * dModel * 4.0;
      double gbps = (totalBytesStep / avgEpochSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = dModel.toString().padRight(6);
      String sAvg = avgEpochMs.toStringAsFixed(2).padRight(12);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sLoss = currentLoss.toStringAsFixed(5);

      print(' $sSeq | $sDim | $sAvg | $sGbps | $sLoss');

      normLayer.free();
      input.free();
      target.free();
      loss.free();
      output.free();
      for (int i = 0; i < intermediates.length; i = i + 1) {
        intermediates[i].free();
      }
    }
  }

  // ===================================================================================================
  // BENCHMARK 2: POSITIONAL ENCODING
  // ===================================================================================================
  print('\n--- 2. POSITIONAL ENCODING BENCHMARK ---');
  print(' SeqLen | dModel | Avg Time(ms) | Throughput (GB/s) | Final Loss');
  print('----------------------------------------------------------------------------------------------------');

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int dModel = dimensions[d];

      List<List<double>> hInput = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add((random.nextDouble() * 2.0) - 1.0);
        }
        hInput.add(row);
      }

      List<List<double>> hTarget = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add(0.5);
        }
        hTarget.add(row);
      }

      GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);
      GPUTensor<Matrix> target = GPUTensor<Matrix>(hTarget);

      // Initialize with seqLength as the maximum supported positions
      PositionalEncodingTL peLayer = PositionalEncodingTL(seqLength, dModel);
      peLayer.build(input);

      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> output = peLayer.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
      GPUTensor<Scalar> loss = GPUTensor<Scalar>(0.0);

      fTape.putInt(OP_MSE_LOSS_FORWARD);
      fTape.putString(output.id);
      fTape.putString(target.id);
      fTape.putString(loss.id);

      loss.creator = GPUNode(
        <GPUTensor>[output, target],
            (CommandBuffer bTape) {
          bTape.putInt(OP_MSE_LOSS_BACKWARD);
          bTape.putString('${output.id}_grad');
          bTape.putString(output.id);
          bTape.putString(target.id);
          bTape.putString('${loss.id}_grad');
        },
        opName: 'mse_loss_manual',
      );

      Uint8List forwardBytes = fTape.bytes();

      CommandBuffer bTape = CommandBuffer();
      SGDGPU optimizer = SGDGPU(peLayer.parameters, learningRate);

      optimizer.zeroGrad(bTape);
      for (int i = 0; i < intermediates.length; i = i + 1) {
        bTape.putInt(OP_ZERO_GRAD);
        bTape.putString('${intermediates[i].id}_grad');
      }
      bTape.putInt(OP_ZERO_GRAD);
      bTape.putString('${input.id}_grad');
      bTape.putInt(OP_ZERO_GRAD);
      bTape.putString('${output.id}_grad');

      bTape.putInt(OP_FILL);
      bTape.putString('${loss.id}_grad');
      bTape.putFloat(1.0);

      loss.backward(bTape);
      Uint8List backwardBytes = bTape.bytes();

      CommandBuffer oTape = CommandBuffer();
      optimizer.step(oTape);
      Uint8List optimizeBytes = oTape.bytes();

      CudaEngine.run(forwardBytes);

      Stopwatch runSw = Stopwatch();
      runSw.start();
      for (int epoch = 1; epoch <= epochsPerTest; epoch = epoch + 1) {
        CudaEngine.run(forwardBytes);
        CudaEngine.run(backwardBytes);

        // Note: Positional Encoding has no learnable weights, so step() does nothing here,
        // but we run it for pipeline consistency.
        CudaEngine.run(optimizeBytes);
      }
      runSw.stop();

      loss.toCpu();
      double currentLoss = loss.value;

      double avgEpochSec = (runSw.elapsedMicroseconds / 1000000.0) / epochsPerTest;
      double avgEpochMs = avgEpochSec * 1000.0;

      // Rough Memory Traffic for PE:
      // Fwd: Read Input, Read PE Matrix, Write Out. Bwd: Read GradOut, Write GradIn.
      // Total approx 5 * (seqLength * dModel * 4 bytes)
      double totalBytesStep = 5.0 * seqLength * dModel * 4.0;
      double gbps = (totalBytesStep / avgEpochSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = dModel.toString().padRight(6);
      String sAvg = avgEpochMs.toStringAsFixed(2).padRight(12);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sLoss = currentLoss.toStringAsFixed(5);

      print(' $sSeq | $sDim | $sAvg | $sGbps | $sLoss');

      peLayer.free();
      input.free();
      target.free();
      loss.free();
      output.free();
      for (int i = 0; i < intermediates.length; i = i + 1) {
        intermediates[i].free();
      }
    }
  }

  print('----------------------------------------------------------------------------------------------------');
  print('Benchmark Complete.');
}