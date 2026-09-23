import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/activationFunction.dart';
import 'layer.dart';

class DenseLayer extends Layer<Vector, Vector> {
  @override
  String name = 'dense';
  int outputSize;
  ActivationFunction<Vector>? activation; // Added <Vector> generic

  late Tensor<Matrix> weights;
  late Tensor<Vector> biases;

  DenseLayer(this.outputSize, {this.activation});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(weights);
    params.add(biases);
    return params;
  }

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

  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    Tensor<Vector> linearOutput = addVector(matVecMul(weights, input), biases);

    if (activation != null) {
      return activation!.call(linearOutput);
    } else {
      return linearOutput;
    }
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['weights'] = weights.value;
    weightsMap['biases'] = biases.value;
    return weightsMap;
  }

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

class DenseLayerMatrix extends Layer<Matrix, Matrix> {
  @override
  String name = 'dense_matrix';
  int outputSize;
  ActivationFunction<Matrix>? activation; // Added <Matrix> generic

  late Tensor<Matrix> weights;
  late Tensor<Vector> biases;

  DenseLayerMatrix(this.outputSize, {this.activation});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(weights);
    params.add(biases);
    return params;
  }

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

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    Tensor<Matrix> linearOutput = addMatrixAndVector(matMul(input, weights), biases);

    if (activation != null) {
      return activation!.call(linearOutput);
    } else {
      return linearOutput;
    }
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['weights'] = weights.value;
    weightsMap['biases'] = biases.value;
    return weightsMap;
  }

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