import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/gpu_version/ffi/OpCodes.dart';
import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/gpu_version/optimizer/SGD.dart';
import 'package:flutter_ml/gpu_version/tapelayertypes/multiHeadAttentionTapeLayer.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';


void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  int numHeads = 8;
  double learningRate = 0.01;
  int epochsPerTest = 10; // Lowered to 10 so you don't wait hours on the massive sizes

  print('====================================================================================================');
  print('                  GPU MULTI-HEAD ATTENTION HEAVYWEIGHT STRESS TEST                                  ');
  print('====================================================================================================');
  print('Heads: $numHeads | LR: $learningRate | Epochs/Test: $epochsPerTest');
  print('----------------------------------------------------------------------------------------------------');
  print(' SeqLen | dModel | Compile(ms) | Avg Time(ms) | Throughput (GB/s) | Compute (TFLOPs) | Final Loss');
  print('----------------------------------------------------------------------------------------------------');

  List<int> sequences = <int>[128, 512, 1024];
  List<int> dimensions = <int>[256, 512, 1024,2048, 4096, 8192];

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int dModel = dimensions[d];

      // 1. Generate Input Matrix
      List<List<double>> hInput = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add((random.nextDouble() * 2.0) - 1.0);
        }
        hInput.add(row);
      }

      // 2. Generate Target Matrix
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

      // 3. Build Layer
      MultiHeadAttentionTL mhaLayer = MultiHeadAttentionTL(dModel, numHeads);
      mhaLayer.build(input);

      Stopwatch compileSw = Stopwatch();
      compileSw.start();

      // ===================================================================
      // FORWARD TAPE
      // ===================================================================
      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> output = mhaLayer.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
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
      // BACKWARD & OPTIMIZER TAPES
      // ===================================================================
      CommandBuffer bTape = CommandBuffer();
      SGDGPU optimizer = SGDGPU(mhaLayer.parameters, learningRate);

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

      compileSw.stop();

      // ===================================================================
      // EXECUTION LOOP & METRIC CALCULATIONS
      // ===================================================================
      // Math for FLOPs (Forward = ~8*S*D^2 + 4*S^2*D. Backward = ~2x Forward)
      double seqD = seqLength.toDouble();
      double modD = dModel.toDouble();
      double headsD = numHeads.toDouble();

      double flopsFwd = (8.0 * seqD * modD * modD) + (4.0 * seqD * seqD * modD);
      double totalFlopsStep = 3.0 * flopsFwd; // Fwd + Bwd

      // Math for Memory Traffic (Bytes Read/Written to VRAM)
      // Estimates base matrices + O(N^2) attention matrices. Bwd is roughly 2x Fwd.
      double bytesFwd = 4.0 * ((3.0 * modD * modD) + (8.0 * seqD * modD) + (2.0 * headsD * seqD * seqD));
      double totalBytesStep = 3.0 * bytesFwd;

      Stopwatch runSw = Stopwatch();

      // Warmup
      CudaEngine.run(forwardBytes);

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

      double tflops = (totalFlopsStep / avgEpochSec) / 1000000000000.0;
      double gbps = (totalBytesStep / avgEpochSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = dModel.toString().padRight(6);
      String sComp = compileSw.elapsedMilliseconds.toString().padRight(11);
      String sAvg = avgEpochMs.toStringAsFixed(1).padRight(12);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sTflops = tflops.toStringAsFixed(4).padRight(16);
      String sLoss = currentLoss.toStringAsFixed(5);

      print(' $sSeq | $sDim | $sComp | $sAvg | $sGbps | $sTflops | $sLoss');

      // Free Memory
      mhaLayer.free();
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