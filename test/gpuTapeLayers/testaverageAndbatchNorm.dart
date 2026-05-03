

import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/gpu_version/tapelayertypes/tapeLayer.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';

List<double> generateVectorData(int size) {
  List<double> data = <double>[];
  for (int i = 0; i < size; i = i + 1) {
    data.add((i % 100).toDouble() / 100.0);
  }
  return data;
}

List<List<double>> generateMatrixData(int rows, int cols) {
  List<List<double>> data = <List<double>>[];
  for (int i = 0; i < rows; i = i + 1) {
    List<double> row = <double>[];
    for (int j = 0; j < cols; j = j + 1) {
      row.add(((i * cols + j) % 100).toDouble() / 100.0);
    }
    data.add(row);
  }
  return data;
}

List<List<List<double>>> generateTensor3DData(int depth, int rows, int cols) {
  List<List<List<double>>> data = [];
  for (int d = 0; d < depth; d = d + 1) {
    List<List<double>> matrix = <List<double>>[];
    for (int i = 0; i < rows; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < cols; j = j + 1) {
        row.add(((d * rows * cols + i * cols + j) % 100).toDouble() / 100.0);
      }
      matrix.add(row);
    }
    data.add(matrix);
  }
  return data;
}

void runBenchmarkGPU(
    String name,
    TapeLayer layer,
    GPUTensor<dynamic> input,
    int iterations,
    double flopsPerIter,
    double bytesPerIter) {
  CommandBuffer tape = CommandBuffer();
  CommandBuffer backwardTape = CommandBuffer();
  List<GPUTensor> intermediates = <GPUTensor>[];

  // Warmup pass
  GPUTensor<dynamic> outWarmup = layer.call(input, tape, intermediates);
  outWarmup.backward(backwardTape);
  CudaEngine.run(tape.bytes());
  CudaEngine.run(backwardTape.bytes());

  outWarmup.free();
  for (int i = 0; i < intermediates.length; i = i + 1) {
    intermediates[i].free();
  }
  intermediates.clear();
  tape.clear();
  backwardTape.clear();

  Stopwatch watch = Stopwatch();
  watch.start();

  for (int i = 0; i < iterations; i = i + 1) {
    GPUTensor<dynamic> out = layer.call(input, tape, intermediates);
    out.backward(backwardTape);

    CudaEngine.run(tape.bytes());
    CudaEngine.run(backwardTape.bytes());

    out.free();
    for (int k = 0; k < intermediates.length; k = k + 1) {
      intermediates[k].free();
    }
    intermediates.clear();
    tape.clear();
    backwardTape.clear();
  }

  watch.stop();

  double totalTimeSec = watch.elapsedMilliseconds / 1000.0;
  double avgTimeMs = watch.elapsedMilliseconds / iterations;
  double avgTimeSec = totalTimeSec / iterations;

  double tflops = (flopsPerIter / avgTimeSec) / 1000000000000.0;
  double gbps = (bytesPerIter / avgTimeSec) / 1000000000.0;

  print('$name:');
  print('  -> Time:  ${avgTimeMs.toStringAsFixed(3)} ms / iter');
  print('  -> Perf:  ${tflops.toStringAsFixed(4)} TFLOP/s');
  print('  -> Mem:   ${gbps.toStringAsFixed(2)} GB/s');
}

void main() {
  CudaEngine.initialize(debug: false);

  print('========================================');
  print('    GPU LAYER SCALING BENCHMARK         ');
  print('========================================\n');

  int iterations = 100;

  // 1. BatchNorm1DGPU
  print('--- BatchNorm1DGPU ---');
  List<int> bn1Sizes = <int>[1024, 8192, 65536, 1048576];
  for (int i = 0; i < bn1Sizes.length; i = i + 1) {
    int size = bn1Sizes[i];
    BatchNorm1DTL bn1 = BatchNorm1DTL(size);
    GPUTensor<Vector> input = GPUTensor<Vector>(generateVectorData(size));

    // Est Flops: Forward (7N), Backward (9N) = 16N
    // Est Bytes: Read/Write weights, inputs, outputs = approx 64 bytes per N
    double flops = (16 * size).toDouble();
    double bytes = (64 * size).toDouble();

    runBenchmarkGPU('Features: $size', bn1, input, iterations, flops, bytes);

    bn1.free();
    input.free();
  }

  // 2. BatchNorm2DGPU
  print('\n--- BatchNorm2DGPU ---');
  List<int> channelsList = <int>[16, 64, 128];
  List<int> spatialList = <int>[64, 128, 256];
  for (int i = 0; i < channelsList.length; i = i + 1) {
    int c = channelsList[i];
    int s = spatialList[i];
    int n = c * s * s;
    BatchNorm2DTL bn2 = BatchNorm2DTL(c);
    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>(generateTensor3DData(c, s, s));

    double flops = (16 * n).toDouble();
    double bytes = ((4 * n + 12 * c) * 4).toDouble();

    runBenchmarkGPU('C: $c, HxW: ${s}x$s', bn2, input, iterations, flops, bytes);

    bn2.free();
    input.free();
  }

  // 3. AveragePooling2DGPU
  print('\n--- AveragePooling2DGPU ---');
  List<int> poolSpatialList = <int>[128, 256, 1024];
  for (int i = 0; i < poolSpatialList.length; i = i + 1) {
    int s = poolSpatialList[i];
    int outS = s ~/ 2;
    int inElements = s * s;
    int outElements = outS * outS;

    AveragePooling2DTL avgPool = AveragePooling2DTL(poolSize: 2, stride: 2);
    GPUTensor<Matrix> input = GPUTensor<Matrix>(generateMatrixData(s, s));

    // Pool 2x2: Fwd (4 adds, 1 div per out), Bwd (1 div, 4 adds per out)
    double flops = (10 * outElements).toDouble();
    // Memory traffic: read in, write out, read gradOut, read/write gradIn
    double bytes = ((3 * inElements + 2 * outElements) * 4).toDouble();

    runBenchmarkGPU('Input: ${s}x$s, Pool: 2', avgPool, input, iterations, flops, bytes);

    avgPool.free();
    input.free();
  }

  // 4. GlobalAveragePoolingGPU
  print('\n--- GlobalAveragePoolingGPU ---');
  List<int> seqList = <int>[64, 256, 1024];
  List<int> dimList = <int>[128, 512, 1024];
  for (int i = 0; i < seqList.length; i = i + 1) {
    int seq = seqList[i];
    int dim = dimList[i];
    GlobalAveragePoolingTL gap = GlobalAveragePoolingTL();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(generateMatrixData(seq, dim));

    // Fwd (adds over seq per dim + div), Bwd (div per dim + adds over seq)
    double flops = (2.0 * dim * seq + 2.0 * dim);
    double bytes = ((3 * seq * dim + 2 * dim) * 4).toDouble();

    runBenchmarkGPU('Seq: $seq, Dim: $dim', gap, input, iterations, flops, bytes);

    gap.free();
    input.free();
  }

  print('\n========================================');
  print('         BENCHMARK COMPLETE             ');
  print('========================================');

  CudaEngine.dispose();
}