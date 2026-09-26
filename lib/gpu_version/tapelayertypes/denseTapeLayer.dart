import 'dart:math';
import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Represents a standard fully connected layer of a neural network.
/// It performs a matrix multiplication and adds a bias vector to the result.
class DenseTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'DenseLayer';

  int outputSize;

  late GPUTensor<Matrix> weights;
  late GPUTensor<Vector> bias;

  /// Stores the batch size of the last batch to see if the allocated memory is fitting
  int cacheBatchSize = -1;
  GPUTensor<Matrix>? matMulResult;
  GPUTensor<Matrix>? cachedOut;

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
  void build(GPUTensor<Matrix> input) {
    int inputSize = input.shape[1];

    Random rnd = Random();
    List<List<double>> wData = <List<double>>[];

    // Simple Xavier initialization scale
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

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);

    built = true;
  }

  /// Appends the matrix multiplication and bias addition to the [tape].
  /// Persistently caches the intermediate multiplication result for static unrolling.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    if (cacheBatchSize != currentBatchSize) {
      if (matMulResult != null) matMulResult!.free();
      if (cachedOut != null) cachedOut!.free();

      matMulResult = null;
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    matMulResult = matMulGPU(input, weights, tape, outTensor: matMulResult);
    cachedOut = addMatrixAndVectorGPU(matMulResult!, bias, tape, outTensor: cachedOut);

    return cachedOut!;
  }

  /// Clears the gradients of all statically cached intermediate tensors to prevent infinite accumulation.
  void zeroStates(CommandBuffer tape) {
    if (matMulResult != null) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${matMulResult!.id}_grad');
    }
    if (cachedOut != null) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${cachedOut!.id}_grad');
    }
  }

  /// Frees VRAM for weights, biases, and the internally cached intermediate tensors.
  @override
  void free() {
    if (built) {
      weights.free();
      bias.free();
    }
    if (matMulResult != null) matMulResult!.free();
    if (cachedOut != null) cachedOut!.free();
  }

  /// Returns [weights] and [bias] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    // Pull from VRAM to CPU before extracting the values
    weights.toCpu();
    bias.toCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  /// Sets [weights] and [bias] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      weights.free();
      bias.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);
    built = true;
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

  GPUTensor<Matrix>? cachedOut;
  GPUTensor<Matrix>? cachedPreRelu;

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
  void build(GPUTensor<Matrix> input) {
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

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);

    built = true;
  }

  /// Appends the fused matrix multiplication, bias addition, and ReLU operation to the [tape].
  /// The fused kernel automatically registers its intermediates.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    // Because the fused kernel allocates internally without an outTensor,
    // we must free the old allocations if forward is called dynamically multiple times.
    if (cachedOut != null) cachedOut!.free();
    if (cachedPreRelu != null) cachedPreRelu!.free();

    List<GPUTensor> localIntermediates = <GPUTensor>[];
    cachedOut = matMulBiasReluGPU(input, weights, bias, tape, localIntermediates);

    if (localIntermediates.isNotEmpty) {
      cachedPreRelu = localIntermediates[0] as GPUTensor<Matrix>;
      intermediates.add(cachedPreRelu!);
    }

    return cachedOut!;
  }

  /// Clears the gradients of all statically cached intermediate tensors to prevent infinite accumulation.
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${cachedOut!.id}_grad');
    }
    if (cachedPreRelu != null) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${cachedPreRelu!.id}_grad');
    }
  }

  /// Frees VRAM for weights, biases, and intermediate tensors.
  @override
  void free() {
    if (built) {
      weights.free();
      bias.free();
    }
    if (cachedOut != null) cachedOut!.free();
    if (cachedPreRelu != null) cachedPreRelu!.free();
  }

  /// Returns [weights] and [bias] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    weights.toCpu();
    bias.toCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  /// Sets [weights] and [bias] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      weights.free();
      bias.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);
    built = true;
  }
}