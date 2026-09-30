import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/activationFunction.dart';
import 'layer.dart';

/// Represents a standard fully connected layer operating on a 1D [Vector] tensor.
/// Performs a matrix-vector multiplication with [weights], adds a [biases] vector,
/// and optionally applies an [activation] function to the resulting vector.
class DenseLayer extends Layer<Vector, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'dense';

  /// Number of output units (neurons) in this layer.
  int outputSize;

  /// Optional activation function applied element-wise to the output vector.
  ActivationFunction<Vector>? activation;

  /// Learnable weight matrix tensor of shape [outputSize, inputSize].
  late Tensor<Matrix> weights;

  /// Learnable bias vector tensor of shape [outputSize].
  late Tensor<Vector> biases;

  /// Creates a [DenseLayer] with the given [outputSize] and an optional [activation] function.
  DenseLayer(this.outputSize, {this.activation});

  /// Returns the trainable [weights] matrix and [biases] vector.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(weights);
    params.add(biases);
    return params;
  }

  /// Allocates and initializes [weights] using a standard normal distribution (Box-Muller)
  /// scaled by He initialization, and sets all [biases] elements to 0.0.
  @override
  void build(Tensor<Vector> input) {
    int inputSize = input.value.length;
    double stddev = sqrt(2.0 / inputSize);
    Random random = Random();

    Matrix w = [];
    for (int i = 0; i < outputSize; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < inputSize; j = j + 1) {
        row.add((sqrt(-2.0 * log(random.nextDouble())) * cos(2.0 * pi * random.nextDouble())) * stddev);
      }
      w.add(row);
    }
    weights = Tensor<Matrix>(w);

    Vector b = [];
    for (int i = 0; i < outputSize; i = i + 1) {
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  /// Computes the linear transformation [weights] * [input] + [biases] on the CPU
  /// using [matVecMul] and [addVector], applying [activation] if defined.
  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    Tensor<Vector> linearOutput = addVector(matVecMul(weights, input), biases);

    if (activation != null) {
      return activation!.call(linearOutput);
    } else {
      return linearOutput;
    }
  }

  /// Returns [weights] and [biases] as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['weights'] = weights.value;
    weightsMap['biases'] = biases.value;
    return weightsMap;
  }

  /// Sets [weights] and [biases] by writing directly to their flat 1D data buffers from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> weightsDynamic = weightsMap['weights'] as List<dynamic>;

    // BUG FIX: Write directly to the 1D flat 'data' array
    int weightIdx = 0;
    for (int i = 0; i < weightsDynamic.length; i = i + 1) {
      List<dynamic> rowDynamic = weightsDynamic[i] as List<dynamic>;
      for (int j = 0; j < rowDynamic.length; j = j + 1) {
        weights.data[weightIdx] = rowDynamic[j] as double;
        weightIdx = weightIdx + 1;
      }
    }

    List<dynamic> biasesDynamic = weightsMap['biases'] as List<dynamic>;

    // BUG FIX: Write directly to the 1D flat 'data' array
    for (int i = 0; i < biasesDynamic.length; i = i + 1) {
      biases.data[i] = biasesDynamic[i] as double;
    }
  }
}

/// Represents a standard fully connected layer operating on a 2D [Matrix] tensor for batched inputs.
/// Performs matrix multiplication of [input] with [weights], broadcasts the [biases] vector across rows,
/// and optionally applies an [activation] function to the output matrix.
class DenseLayerMatrix extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'dense_matrix';

  /// Number of output units (features per sample) produced by this layer.
  int outputSize;

  /// Optional activation function applied element-wise to the output matrix.
  ActivationFunction<Matrix>? activation;

  /// Learnable weight matrix tensor of shape [inputSize, outputSize].
  late Tensor<Matrix> weights;

  /// Learnable bias vector tensor of shape [outputSize].
  late Tensor<Vector> biases;

  /// Creates a [DenseLayerMatrix] with the given [outputSize] and an optional [activation] function.
  DenseLayerMatrix(this.outputSize, {this.activation});

  /// Returns the trainable [weights] matrix and [biases] vector.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(weights);
    params.add(biases);
    return params;
  }

  /// Allocates and initializes [weights] using a standard normal distribution (Box-Muller)
  /// scaled by He initialization, and sets all [biases] elements to 0.0.
  @override
  void build(Tensor<Matrix> input) {
    int inputSize = input.value[0].length;
    double stddev = sqrt(2.0 / inputSize);
    Random random = Random();

    Matrix w = [];
    for (int i = 0; i < inputSize; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add((sqrt(-2.0 * log(random.nextDouble())) * cos(2.0 * pi * random.nextDouble())) * stddev);
      }
      w.add(row);
    }
    weights = Tensor<Matrix>(w);

    Vector b = [];
    for (int i = 0; i < outputSize; i = i + 1) {
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  /// Computes the batched linear transformation [input] * [weights] + [biases] on the CPU
  /// using [matMul] and [addMatrixAndVector], applying [activation] if defined.
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    Tensor<Matrix> linearOutput = addMatrixAndVector(matMul(input, weights), biases);

    if (activation != null) {
      return activation!.call(linearOutput);
    } else {
      return linearOutput;
    }
  }

  /// Returns [weights] and [biases] as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['weights'] = weights.value;
    weightsMap['biases'] = biases.value;
    return weightsMap;
  }

  /// Sets [weights] and [biases] by writing directly to their flat 1D data buffers from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> weightsDynamic = weightsMap['weights'] as List<dynamic>;

    // BUG FIX: Write directly to the 1D flat 'data' array
    int weightIdx = 0;
    for (int i = 0; i < weightsDynamic.length; i = i + 1) {
      List<dynamic> rowDynamic = weightsDynamic[i] as List<dynamic>;
      for (int j = 0; j < rowDynamic.length; j = j + 1) {
        weights.data[weightIdx] = rowDynamic[j] as double;
        weightIdx = weightIdx + 1;
      }
    }

    List<dynamic> biasesDynamic = weightsMap['biases'] as List<dynamic>;

    // BUG FIX: Write directly to the 1D flat 'data' array
    for (int i = 0; i < biasesDynamic.length; i = i + 1) {
      biases.data[i] = biasesDynamic[i] as double;
    }
  }
}