import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A standard, fully-connected neural network layer for 1D Vector data.
///
/// A `DenseLayer` implements the operation: `activation(weights @ input + biases)`.
/// It is the most common layer for processing flat feature vectors.
///
/// - **Input:** A `Tensor<Vector>` of shape `[input_size]`.
/// - **Output:** A `Tensor<Vector>` of shape `[output_size]`.
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
    for(int i=0; i<outputSize; i++){
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Tensor<Vector> linearOutput = add(matVecMul(weights, input as Tensor<Vector>), biases);

    if (activation != null) {
      return activation!.call(linearOutput) as Tensor<Vector>;
    } else {
      return linearOutput;
    }
  }
}

/// A fully-connected layer that operates on a batch of data (a Matrix).
///
/// A `DenseLayerMatrix` applies the same dense transformation to every row
/// of the input matrix. It implements the operation: `activation(input @ weights + biases)`.
///
/// This is used for batch processing or in architectures like Transformers where
/// sequences of vectors are processed.
///
/// - **Input:** A `Tensor<Matrix>` of shape `[batch_size, input_size]`.
/// - **Output:** A `Tensor<Matrix>` of shape `[batch_size, output_size]`.
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
    for(int i=0; i<outputSize; i++){
      b.add(0.0);
    }
    biases = Tensor<Vector>(b);

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> linearOutput = addMatrixAndVector(matMul(input as Tensor<Matrix>, weights), biases);

    if (activation != null) {
      // Assumes your activation function can handle matrices (e.g., using reluMatrix)
      return activation!.call(linearOutput) as Tensor<Matrix>;
    } else {
      return linearOutput;
    }
  }
}