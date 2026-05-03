import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/full_library.dart';

void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  int numHeads = 8;
  int runsPerTest = 50;

  print('====================================================================================================');
  print('              GPU TRANSFORMER ENCODER BLOCK INFERENCE BENCHMARK (FORWARD ONLY)                      ');
  print('====================================================================================================');
  print('Heads: $numHeads | Feed-Forward Ratio: 4x dModel | Inference Runs/Test: $runsPerTest');
  print('----------------------------------------------------------------------------------------------------');
  print(' SeqLen | dModel | Compile(ms) | Latency(ms) | Throughput (GB/s) | Compute (TFLOPs)');
  print('----------------------------------------------------------------------------------------------------');

  List<int> sequences = <int>[128, 512, 1024, 2048];
  List<int> dimensions = <int>[256, 1024, 2048, 4096,8192];

  for (int s = 0; s < sequences.length; s = s + 1) {
    int seqLength = sequences[s];

    for (int d = 0; d < dimensions.length; d = d + 1) {
      int dModel = dimensions[d];
      int dff = dModel * 4; // Standard Transformer configuration

      // 1. Generate Input Matrix (Batch = 1)
      List<List<double>> hInput = <List<double>>[];
      for (int i = 0; i < seqLength; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < dModel; j = j + 1) {
          row.add((random.nextDouble() * 2.0) - 1.0);
        }
        hInput.add(row);
      }

      GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);

      // 2. Build Transformer Block
      TransformerEncoderBlockTapeLayer transformerBlock = TransformerEncoderBlockTapeLayer(dModel, numHeads, dff);
      transformerBlock.build(input);

      Stopwatch compileSw = Stopwatch();
      compileSw.start();

      // ===================================================================
      // FORWARD TAPE (INFERENCE ONLY)
      // ===================================================================
      CommandBuffer fTape = CommandBuffer();
      List<GPUTensor> intermediates = <GPUTensor>[];

      GPUTensor<Matrix> output = transformerBlock.forward(input, fTape, intermediates) as GPUTensor<Matrix>;

      Uint8List forwardBytes = fTape.bytes();

      compileSw.stop();

      // ===================================================================
      // EXECUTION LOOP & METRIC CALCULATIONS
      // ===================================================================

      double seqD = seqLength.toDouble();
      double modD = dModel.toDouble();
      double ffD = dff.toDouble();

      // FLOPs purely for the forward pass
      // MHA (Proj + Out): 8 * Seq * D^2
      // MHA (Attention):  4 * Seq^2 * D
      // FFN (W1 + W2):    4 * Seq * D * FF
      double totalFlopsStep = (8.0 * seqD * modD * modD) +
          (4.0 * seqD * seqD * modD) +
          (4.0 * seqD * modD * ffD);

      // Memory Traffic (Bytes Read/Written to VRAM)
      // Weights: ~4D^2 (MHA) + 2*D*FF (FFN) -> * 4 bytes
      double weightBytes = (4.0 * modD * modD + 2.0 * modD * ffD) * 4.0;
      // Activations: Rough estimate of intermediate reads/writes per step
      double actBytes = ((20.0 * seqD * modD) + (5.0 * seqD * ffD) + (4.0 * seqD * seqD)) * 4.0;
      double totalBytesStep = weightBytes + actBytes;

      Stopwatch runSw = Stopwatch();

      // Warmup (Push weights into VRAM caches)
      CudaEngine.run(forwardBytes);

      // Inference Loop
      runSw.start();
      for (int run = 1; run <= runsPerTest; run = run + 1) {
        CudaEngine.run(forwardBytes);
      }
      runSw.stop();

      double avgRunSec = (runSw.elapsedMicroseconds / 1000000.0) / runsPerTest;
      double avgRunMs = avgRunSec * 1000.0;

      double tflops = (totalFlopsStep / avgRunSec) / 1000000000000.0;
      double gbps = (totalBytesStep / avgRunSec) / 1000000000.0;

      String sSeq = seqLength.toString().padRight(6);
      String sDim = dModel.toString().padRight(6);
      String sComp = compileSw.elapsedMilliseconds.toString().padRight(11);
      String sAvg = avgRunMs.toStringAsFixed(2).padRight(11);
      String sGbps = gbps.toStringAsFixed(2).padRight(17);
      String sTflops = tflops.toStringAsFixed(4).padRight(16);

      print(' $sSeq | $sDim | $sComp | $sAvg | $sGbps | $sTflops');

      // Free Memory
      transformerBlock.free();
      input.free();
      output.free();
      for (int i = 0; i < intermediates.length; i = i + 1) {
        intermediates[i].free();
      }
    }
  }

  print('----------------------------------------------------------------------------------------------------');
  print('Benchmark Complete.');
}