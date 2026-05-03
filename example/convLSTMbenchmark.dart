import 'dart:typed_data';

import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/gpu_version/tapelayertypes/convLSTMTapeLayer.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';


void main() async{
  await CudaEngine.initialize(debug: false);

  int height = 256;
  int width = 256;
  int hiddenFilters = 8;
  int kernelSize = 3;

  print('Starting ConvLSTM FORWARD Benchmark (HEAVY LOAD - MANUAL RUN)');
  print('Config: [${height}x$width], Filters=$hiddenFilters, Kernel=$kernelSize');
  print('---------------------------------------------------------------------------------------------------------');
  print('Seq Len\t| Predict Avg (ms)\t| Throughput (S/sec)\t| Compute (TFLOPS)\t| Bandwidth (GB/s)');
  print('---------------------------------------------------------------------------------------------------------');

  for (int size = 4; size <= 128; size = size * 2) {
    int seqLength = size;

    List<List<List<double>>> hInput = <List<List<double>>>[];
    for (int s = 0; s < seqLength; s = s + 1) {
      List<List<double>> matrix = <List<double>>[];
      for (int i = 0; i < height; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < width; j = j + 1) {
          row.add(1.0);
        }
        matrix.add(row);
      }
      hInput.add(matrix);
    }

    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>(hInput);

    ConvLSTMTL layer = ConvLSTMTL(hiddenFilters, kernelSize);
    layer.build(input);

    CommandBuffer tape = CommandBuffer();
    List<GPUTensor> intermediates = <GPUTensor>[];

    GPUTensor<dynamic> outDyn = layer.forward(input, tape, intermediates);
    GPUTensor<Tensor3D> out = outDyn as GPUTensor<Tensor3D>;

    Uint8List compiledTape = tape.bytes();

    // Warm-up run
    CudaEngine.run(compiledTape);

    int iterations = 20;
    Stopwatch sw = Stopwatch();

    sw.start();
    for (int i = 0; i < iterations; i = i + 1) {
      CudaEngine.run(compiledTape);
    }
    sw.stop();
    //TapeDecoder(compiledTape).decode();

    double avgPredictMs = (sw.elapsedMicroseconds / 1000.0) / iterations;
    double avgPredictSec = avgPredictMs / 1000.0;
    double throughput = 1.0 / avgPredictSec;

    double stepFlops = (height * width) * (16.0 * kernelSize * kernelSize + 17.0);
    double totalFlops = seqLength * stepFlops;
    double tflops = (totalFlops / avgPredictSec) / 1e12;

    double stepBytes = 62.0 * height * width * 4.0;
    double totalBytes = seqLength * stepBytes;
    double gbPerSec = (totalBytes / avgPredictSec) / 1e9;

    String sSize = size.toString().padRight(7);
    String sInf = avgPredictMs.toStringAsFixed(3).padRight(22);
    String sThroughput = throughput.toStringAsFixed(2).padRight(24);
    String sTflops = tflops.toStringAsFixed(4).padRight(22);
    String sGbps = gbPerSec.toStringAsFixed(2);

    print('$sSize| $sInf| $sThroughput| $sTflops| $sGbps');

    input.free();
    layer.free();
    out.free();
    for (int i = 0; i < intermediates.length; i = i + 1) {
      intermediates[i].free();
    }
  }

  print('---------------------------------------------------------------------------------------------------------');
  print('Benchmark Complete.');
}