import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../activationFunctions/relu.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

class DenseLayer extends Layer {
  @override
  String name = 'dense';
  int outputSize;
  ActivationFunction? activation;

  late Tensor<Matrix> weights;
  late Tensor<Vector> biases;

  DenseLayer(this.outputSize, {this.activation});

  @override
  List<Tensor> get parameters => [weights, biases];

  @override
  void build(Tensor<dynamic> input) {
    int inputSize = (input.value as Vector).length;
    double stddev = sqrt(2.0 / inputSize);
    Random random = Random();

    Matrix w = [];
    for (int i = 0; i < outputSize; i++) {
      Vector row = [];
      for (int j = 0; j < inputSize; j++) {
        row.add((sqrt(-2 * log(random.nextDouble())) * cos(2 * pi * random.nextDouble())) * stddev);
      }
      w.add(row);
    }
    weights = Tensor<Matrix>(w);

    Vector b = [];
    for(int i = 0; i < outputSize; i++){
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Tensor<Vector> linearOutput = addVector(matVecMul(weights, input as Tensor<Vector>), biases);

    if (activation != null) {
      return activation!.call(linearOutput) as Tensor<Vector>;
    } else {
      return linearOutput;
    }
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'weights': weights.value,
      'biases': biases.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> weightsDynamic = weightsMap['weights'] as List<dynamic>;
    Matrix newWeights = weightsDynamic.map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();

    Vector newBiases = (weightsMap['biases'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < weights.value.length; i++) {
      for (int j = 0; j < weights.value[0].length; j++) {
        weights.value[i][j] = newWeights[i][j];
      }
    }

    for (int i = 0; i < biases.value.length; i++) {
      biases.value[i] = newBiases[i];
    }
  }
}

class DenseLayerMatrix extends Layer {
  @override
  String name = 'dense_matrix';
  int outputSize;
  ActivationFunction? activation;

  late Tensor<Matrix> weights;
  late Tensor<Vector> biases;

  DenseLayerMatrix(this.outputSize, {this.activation});

  @override
  List<Tensor> get parameters => [weights, biases];

  @override
  void build(Tensor<dynamic> input) {
    int inputSize = (input.value as Matrix)[0].length;
    double stddev = sqrt(2.0 / inputSize);
    Random random = Random();

    Matrix w = [];
    for (int i = 0; i < inputSize; i++) {
      Vector row = [];
      for (int j = 0; j < outputSize; j++) {
        row.add((sqrt(-2 * log(random.nextDouble())) * cos(2 * pi * random.nextDouble())) * stddev);
      }
      w.add(row);
    }
    weights = Tensor<Matrix>(w);

    Vector b = [];
    for(int i = 0; i < outputSize; i++){
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> linearOutput = addMatrixAndVector(matMul(input as Tensor<Matrix>, weights), biases);

    if (activation != null) {
      return activation!.call(linearOutput) as Tensor<Matrix>;
    } else {
      return linearOutput;
    }
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'weights': weights.value,
      'biases': biases.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> weightsDynamic = weightsMap['weights'] as List<dynamic>;
    Matrix newWeights = weightsDynamic.map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();

    Vector newBiases = (weightsMap['biases'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < weights.value.length; i++) {
      for (int j = 0; j < weights.value[0].length; j++) {
        weights.value[i][j] = newWeights[i][j];
      }
    }

    for (int i = 0; i < biases.value.length; i++) {
      biases.value[i] = newBiases[i];
    }
  }
}