import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/full_library.dart';

void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  int numHeads = 8;
  double learningRate = 0.01;
  int runsPerTest = 20;

  print('====================================================================================================');
  print('             GPU TRANSFORMER ENCODER BLOCK FULL TRAINING BENCHMARK (FWD + BWD + OPT)                ');
  print('====================================================================================================');
  print('Heads: $numHeads | FF-Ratio: 4x dModel | LR: $learningRate | Training Steps/Test: $runsPerTest');
  print('----------------------------------------------------------------------------------------------------');
  print(' SeqLen | dModel | Compile(ms) | Step Time(ms)| Throughput (GB/s) | Compute (TFLOPs) | Final Loss');
  print('----------------------------------------------------------------------------------------------------');

  // Scaled down slightly to prevent RTX 3060 VRAM OOM during the backward pass memory hoarding
  List<int> sequences = <int>[128, 512, 1024];
  List<int> dimensions = <int>[256, 1024, 2048, 4096,8192];

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int dModel = dimensions[d];
      int dff = dModel * 4;

      // 1. Generate Input & Target
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

      // 2. Build Transformer Block
      TransformerEncoderBlockTapeLayer transformerBlock = TransformerEncoderBlockTapeLayer(dModel, numHeads, dff);
      transformerBlock.build(input);

      Stopwatch compileSw = Stopwatch();
      compileSw.start();

      // ===================================================================
      // FORWARD TAPE
      // ===================================================================
      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> output = transformerBlock.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
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

      // ===================================================================
      // BACKWARD TAPE
      // ===================================================================
      CommandBuffer bTape = CommandBuffer();
      SGDGPU optimizer = SGDGPU(transformerBlock.parameters, learningRate);

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

      // ===================================================================
      // OPTIMIZER TAPE
      // ===================================================================
      CommandBuffer oTape = CommandBuffer();
      optimizer.step(oTape);
      Uint8List optimizeBytes = oTape.bytes();

      compileSw.stop();

      // ===================================================================
      // METRIC CALCULATIONS (FWD + BWD)
      // ===================================================================
      double seqD = seqLength.toDouble();
      double modD = dModel.toDouble();
      double ffD = dff.toDouble();

      // Compute: Backward is roughly 2x Forward. Total = 3x Fwd.
      double flopsFwd = (8.0 * seqD * modD * modD) + (4.0 * seqD * seqD * modD) + (4.0 * seqD * modD * ffD);
      double totalFlopsStep = 3.0 * flopsFwd;

      // Memory: Rough heuristic. Fwd + Bwd + Opt traffic is roughly 3x the pure inference traffic.
      double weightBytes = (4.0 * modD * modD + 2.0 * modD * ffD) * 4.0;
      double actBytes = ((20.0 * seqD * modD) + (5.0 * seqD * ffD) + (4.0 * seqD * seqD)) * 4.0;
      double totalBytesStep = 3.0 * (weightBytes + actBytes);

      Stopwatch runSw = Stopwatch();

      // Warmup
      CudaEngine.run(forwardBytes);

      // Full Training Loop
      runSw.start();
      for (int run = 1; run <= runsPerTest; run = run + 1) {
        CudaEngine.run(forwardBytes);
        CudaEngine.run(backwardBytes);
        CudaEngine.run(optimizeBytes);
      }
      runSw.stop();

      loss.toCpu();
      double currentLoss = loss.value;

      double avgRunSec = (runSw.elapsedMicroseconds / 1000000.0) / runsPerTest;
      double avgRunMs = avgRunSec * 1000.0;

      double tflops = (totalFlopsStep / avgRunSec) / 1000000000000.0;
      double gbps = (totalBytesStep / avgRunSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = dModel.toString().padRight(6);
      String sComp = compileSw.elapsedMilliseconds.toString().padRight(11);
      String sAvg = avgRunMs.toStringAsFixed(2).padRight(13);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sTflops = tflops.toStringAsFixed(4).padRight(16);
      String sLoss = currentLoss.toStringAsFixed(5);

      print(' $sSeq | $sDim | $sComp | $sAvg | $sGbps | $sTflops | $sLoss');

      // Free Memory
      transformerBlock.free();
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