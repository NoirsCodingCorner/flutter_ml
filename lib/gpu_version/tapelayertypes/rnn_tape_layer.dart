import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// A standard Recurrent Neural Network (RNN) Layer.
/// Processes a sequence of data sequentially, maintaining a hidden state that is updated at each time step
/// based on the current input and the previous hidden state.
class RNNTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'RNNTapeLayer';

  int hiddenSize;
  String activation;

  /// Learnable weight matrix transforming the current input.
  late GPUTensor<Matrix> wXh;

  /// Learnable weight matrix transforming the previous hidden state.
  late GPUTensor<Matrix> wHh;

  /// Learnable bias column matrix added to the combined input and hidden state.
  late GPUTensor<Matrix> bh;

  /// Requires the size of the hidden state [hiddenSize].
  /// The [activation] can be either 'relu' or 'tanh'.
  RNNTL(this.hiddenSize, {this.activation = 'relu'});

  /// Returns the trainable weight matrices and bias.
  /// Specifically [wXh],[wHh] and [bh].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(wXh);
      params.add(wHh);
      params.add(bh);
    }
    return params;
  }

  /// Allocates VRAM for weights and biases using a standard Xavier initialization.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    int inputSize = input.shape[1];

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

    wXh = GPUTensor<Matrix>(wXhValues, allocTape, sourceOp: '$name:wXh');
    wHh = GPUTensor<Matrix>(wHhValues, allocTape, sourceOp: '$name:wHh');

    List<List<double>> bHValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      bHValues.add(<double>[0.0]);
    }
    bh = GPUTensor<Matrix>(bHValues, allocTape, sourceOp: '$name:bh');

    built = true;
  }

  /// Appends the recurrent step-by-step operations to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    int seqLength = input.shape[0];
    List<int> stateShape = <int>[hiddenSize, 1];

    List<GPUTensor<Matrix>> hStates = <GPUTensor<Matrix>>[];
    hStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initH'));

    GPUTensor<Matrix> transposedInput = transposeGPU(input, tape, allocTape);

    for (int i = 0; i < seqLength; i = i + 1) {
      GPUTensor<Matrix> hPrev = hStates[i];

      GPUTensor<Matrix> xT =
          sliceColumnGPU(transposedInput, i, i + 1, tape, allocTape);

      GPUTensor<Matrix> inputPart = matMulGPU(wXh, xT, tape, allocTape);

      GPUTensor<Matrix> hiddenPart = matMulGPU(wHh, hPrev, tape, allocTape);

      GPUTensor<Matrix> sum1 =
          addMatrixGPU(inputPart, hiddenPart, tape, allocTape);

      GPUTensor<Matrix> combined = addMatrixGPU(sum1, bh, tape, allocTape);

      GPUTensor<Matrix> newH;
      if (activation == 'relu') {
        newH = reluMatrixGPU(combined, tape, allocTape);
      } else {
        newH = tanhMatrixGPU(combined, tape, allocTape);
      }

      hStates.add(newH);
    }

    return hStates.last;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    wXh.dataToCpu();
    wHh.dataToCpu();
    bh.dataToCpu();

    wMap['W_xh'] = wXh.value;
    wMap['W_hh'] = wHh.value;
    wMap['b_h'] = bh.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> extractFlatMatrix(List<dynamic> raw) {
      List<double> flat = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          flat.add(rawRow[j] as double);
        }
      }
      return flat;
    }

    wXh.pushData(extractFlatMatrix(newWeights['W_xh']!));
    wHh.pushData(extractFlatMatrix(newWeights['W_hh']!));
    bh.pushData(extractFlatMatrix(newWeights['b_h']!));
  }
}
