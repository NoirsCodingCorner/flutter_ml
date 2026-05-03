import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/full_library.dart';




void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  double learningRate = 0.01;
  int runsPerTest = 20;

  print('====================================================================================================');
  print('                    GPU RNN FULL TRAINING SCALING BENCHMARK (UNIFIED TAPE)                          ');
  print('====================================================================================================');
  print('Activation: Tanh | LR: $learningRate | Training Steps/Test: $runsPerTest');
  print('----------------------------------------------------------------------------------------------------');
  print(' SeqLen | dModel | Compile(ms) | Step Time(ms)| Throughput (GB/s) | Compute (TFLOPs) | Final Loss');
  print('----------------------------------------------------------------------------------------------------');

  List<int> sequences = <int>[16, 64, 256];
  List<int> dimensions = <int>[256, 1024, 2048, 4096];

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int hiddenSize = dimensions[d];
      int inputSize = hiddenSize;

      // 1. Generate Input Sequence
      List<List<double>> hInput = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < inputSize; j = j + 1) {
          row.add((random.nextDouble() * 2.0) - 1.0);
        }
        hInput.add(row);
      }

      // 2. Generate Target Matrix [HiddenSize, 1]
      List<List<double>> hTarget = <List<double>>[];
      for (int i = 0; i < hiddenSize; i = i + 1) {
        double val = (i % 2 == 0) ? 0.5 : -0.5;
        hTarget.add(<double>[val]);
      }

      GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);
      GPUTensor<Matrix> target = GPUTensor<Matrix>(hTarget);

      // 3. Build Layer
      RNNTL rnn = RNNTL(hiddenSize, activation: 'tanh');
      rnn.build(input);

      Stopwatch compileSw = Stopwatch();
      compileSw.start();

      // ===================================================================
      // FORWARD TAPE
      // ===================================================================
      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> finalHiddenState = rnn.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
      GPUTensor<Scalar> loss = mseMatrixGPU(finalHiddenState, target, fTape);

      Uint8List forwardBytes = fTape.bytes();

      // ===================================================================
      // BACKWARD TAPE
      // ===================================================================
      CommandBuffer bTape = CommandBuffer();
      SGDGPU optimizer = SGDGPU(rnn.parameters, learningRate);

      optimizer.zeroGrad(bTape);
      for (int i = 0; i < intermediates.length; i = i + 1) {
        bTape.putInt(OP_ZERO_GRAD);
        bTape.putString('${intermediates[i].id}_grad');
      }
      bTape.putInt(OP_ZERO_GRAD);
      bTape.putString('${input.id}_grad');

      loss.backward(bTape);

      List<GPUTensor> params = rnn.parameters;
      for (int i = 0; i < params.length; i = i + 1) {
        bTape.putInt(OP_CLIP_GRAD_VALUE);
        bTape.putString('${params[i].id}_grad');
        bTape.putFloat(1.0);
      }

      Uint8List backwardBytes = bTape.bytes();

      // ===================================================================
      // OPTIMIZER TAPE
      // ===================================================================
      CommandBuffer oTape = CommandBuffer();
      optimizer.step(oTape);
      Uint8List optimizeBytes = oTape.bytes();

      // ⚡ UNIFIED TAPE CONCATENATION ⚡
      int totalLength = forwardBytes.length + backwardBytes.length + optimizeBytes.length;
      Uint8List unifiedTape = Uint8List(totalLength);

      int offset = 0;
      for (int i = 0; i < forwardBytes.length; i = i + 1) {
        unifiedTape[offset] = forwardBytes[i];
        offset = offset + 1;
      }
      for (int i = 0; i < backwardBytes.length; i = i + 1) {
        unifiedTape[offset] = backwardBytes[i];
        offset = offset + 1;
      }
      for (int i = 0; i < optimizeBytes.length; i = i + 1) {
        unifiedTape[offset] = optimizeBytes[i];
        offset = offset + 1;
      }

      compileSw.stop();

      // ===================================================================
      // METRIC CALCULATIONS
      // ===================================================================
      double seqD = seqLength.toDouble();
      double inD = inputSize.toDouble();
      double hidD = hiddenSize.toDouble();

      // FLOPs purely for the forward pass per step: W_xh * x_t + W_hh * h_prev
      double flopsFwd = seqD * ((2.0 * inD * hidD) + (2.0 * hidD * hidD));
      double totalFlopsStep = 3.0 * flopsFwd; // Fwd + roughly 2x for Bwd

      // Memory Traffic (Bytes Read/Written to VRAM)
      double weightBytes = ((inD * hidD) + (hidD * hidD) + hidD) * 4.0;
      double actBytes = seqD * (inD + (5.0 * hidD)) * 4.0;
      double totalBytesStep = 3.0 * (weightBytes + actBytes); // Opt touches weights again

      Stopwatch runSw = Stopwatch();

      // Warmup (Push weights into VRAM caches)
      CudaEngine.run(unifiedTape);

      // Main Loop
      runSw.start();
      for (int run = 1; run <= runsPerTest; run = run + 1) {
        CudaEngine.run(unifiedTape);
      }
      runSw.stop();

      loss.toCpu();
      double currentLoss = loss.value;

      double avgRunSec = (runSw.elapsedMicroseconds / 1000000.0) / runsPerTest;
      double avgRunMs = avgRunSec * 1000.0;

      double tflops = (totalFlopsStep / avgRunSec) / 1000000000000.0;
      double gbps = (totalBytesStep / avgRunSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = hiddenSize.toString().padRight(6);
      String sComp = compileSw.elapsedMilliseconds.toString().padRight(11);
      String sAvg = avgRunMs.toStringAsFixed(2).padRight(13);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sTflops = tflops.toStringAsFixed(4).padRight(16);
      String sLoss = currentLoss.toStringAsFixed(5);

      print(' $sSeq | $sDim | $sComp | $sAvg | $sGbps | $sTflops | $sLoss');

      // Free Memory
      rnn.free();
      input.free();
      target.free();
      loss.free();
      finalHiddenState.free();
      for (int i = 0; i < intermediates.length; i = i + 1) {
        intermediates[i].free();
      }
    }
  }

  print('----------------------------------------------------------------------------------------------------');
  print('Benchmark Complete.');
}