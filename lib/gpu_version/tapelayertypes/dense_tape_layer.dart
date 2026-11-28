import 'dart:math';
import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Represents a standard fully connected layer of a neural network.
/// It performs a matrix multiplication and adds a bias vector to the result.
class DenseTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'DenseLayer';

  int outputSize;

  late GPUTensor<Matrix> weights;
  late GPUTensor<Vector> bias;

  /// Requires the [outputSize] which determines the amount of neurons in this layer.
  DenseTL(this.outputSize);

  /// Returns the trainable [weights] and [bias].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(bias);
    }
    return params;
  }

  /// Allocates VRAM for weights and biases using a standard Xavier initialization.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    int inputSize = input.shape[1];

    Random rnd = Random();
    List<List<double>> wData = <List<double>>[];

    double scale = sqrt(2.0 / (inputSize + outputSize));

    for (int i = 0; i < inputSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add((rnd.nextDouble() * 2.0 - 1.0) * scale);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    for (int i = 0; i < outputSize; i = i + 1) {
      bData.add(0.0);
    }

    weights = GPUTensor<Matrix>(wData, allocTape, sourceOp: '$name:weights');
    bias = GPUTensor<Vector>(bData, allocTape, sourceOp: '$name:bias');

    built = true;
  }

  /// Appends the matrix multiplication and bias addition to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> matMulResult = matMulGPU(input, weights, tape, allocTape);
    GPUTensor<Matrix> out =
    addMatrixAndVectorGPU(matMulResult, bias, tape, allocTape);

    return out;
  }

  /// Returns [weights] and [bias] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    weights.dataToCpu();
    bias.dataToCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  /// Sets [weights] and [bias] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> flatWeights = <double>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        flatWeights.add(rawRow[j] as double);
      }
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    weights.pushData(flatWeights);
    bias.pushData(bData);
  }
}

/// Represents a standard fully connected layer followed by a ReLU activation function.
/// Utilizes a fused GPU kernel to perform matrix multiplication, bias broadcasting,
/// and ReLU activation in a single optimized pass.
class DenseReluTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'DenseReluLayer';

  int outputSize;

  late GPUTensor<Matrix> weights;
  late GPUTensor<Vector> bias;

  /// Requires the [outputSize] which determines the amount of neurons in this layer.
  DenseReluTL(this.outputSize);

  /// Returns the trainable [weights] and [bias].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(bias);
    }
    return params;
  }

  /// Allocates VRAM for weights and biases using a standard Xavier initialization.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    int inputSize = input.shape[1];

    Random rnd = Random();
    List<List<double>> wData = <List<double>>[];

    double scale = sqrt(2.0 / (inputSize + outputSize));

    for (int i = 0; i < inputSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add((rnd.nextDouble() * 2.0 - 1.0) * scale);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    for (int i = 0; i < outputSize; i = i + 1) {
      bData.add(0.0);
    }

    weights = GPUTensor<Matrix>(wData, allocTape, sourceOp: '$name:weights');
    bias = GPUTensor<Vector>(bData, allocTape, sourceOp: '$name:bias');

    built = true;
  }

  /// Appends the fused matrix multiplication, bias addition, and ReLU operation to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> out =
    matMulBiasReluGPU(input, weights, bias, tape, allocTape);

    return out;
  }

  /// Returns [weights] and [bias] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    weights.dataToCpu();
    bias.dataToCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  /// Sets [weights] and [bias] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> flatWeights = <double>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        flatWeights.add(rawRow[j] as double);
      }
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    weights.pushData(flatWeights);
    bias.pushData(bData);
  }
}