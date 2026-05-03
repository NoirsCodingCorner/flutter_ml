import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class RNNTL extends TapeLayer {
  @override
  String get name {
    return 'RNNTapeLayer';
  }

  int hiddenSize;
  String activation;

  late GPUTensor<Matrix> W_xh;
  late GPUTensor<Matrix> W_hh;
  late GPUTensor<Matrix> b_h;

  RNNTL(this.hiddenSize, {this.activation = 'relu'});

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(W_xh);
      params.add(W_hh);
      params.add(b_h);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    int inputSize = typedInput.shape[1];

    Random random = Random();
    double xavierStdDev(int fanIn, int fanOut) {
      return sqrt(2.0 / (fanIn + fanOut));
    }

    double inputToHiddenStdDev = xavierStdDev(inputSize, hiddenSize);
    List<List<double>> wXhValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < inputSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * inputToHiddenStdDev);
      }
      wXhValues.add(row);
    }

    double hiddenToHiddenStdDev = xavierStdDev(hiddenSize, hiddenSize);
    List<List<double>> wHhValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < hiddenSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * hiddenToHiddenStdDev);
      }
      wHhValues.add(row);
    }

    W_xh = GPUTensor<Matrix>(wXhValues);
    W_hh = GPUTensor<Matrix>(wHhValues);

    List<List<double>> bHValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      bHValues.add(<double>[0.0]);
    }
    b_h = GPUTensor<Matrix>(bHValues);

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    int totalSteps = typedInput.shape[0];

    // Transpose the input so we can slice robust 2D columns instead of fragile 1D rows
    GPUTensor<Matrix> transposedInput = transposeGPU(typedInput, tape);
    intermediates.add(transposedInput);

    List<List<double>> initialHValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      initialHValues.add(<double>[0.0]);
    }

    GPUTensor<Matrix> h = GPUTensor<Matrix>(initialHValues);
    intermediates.add(h);

    for (int i = 0; i < totalSteps; i = i + 1) {
      GPUTensor<Matrix> hPrev = h;

      // Extract time step as a strict [InputSize, 1] Matrix
      GPUTensor<Matrix> x_t = sliceColumnGPU(transposedInput, i, i + 1, tape);
      intermediates.add(x_t);

      GPUTensor<Matrix> inputPart = matMulGPU(W_xh, x_t, tape);
      intermediates.add(inputPart);

      GPUTensor<Matrix> hiddenPart = matMulGPU(W_hh, hPrev, tape);
      intermediates.add(hiddenPart);

      GPUTensor<Matrix> sum1 = addMatrixGPU(inputPart, hiddenPart, tape);
      intermediates.add(sum1);

      GPUTensor<Matrix> combined = addMatrixGPU(sum1, b_h, tape);
      intermediates.add(combined);

      if (activation == 'relu') {
        h = reluMatrixGPU(combined, tape);
      } else {
        h = tanhMatrixGPU(combined, tape);
      }
      intermediates.add(h);
    }

    // Return the pure 2D Matrix representation of the hidden state
    return h;
  }

  @override
  void free() {
    if (built) {
      W_xh.free();
      W_hh.free();
      b_h.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}