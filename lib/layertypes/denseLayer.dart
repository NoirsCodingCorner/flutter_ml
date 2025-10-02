import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A standard, fully-connected neural network layer.
///
/// A `DenseLayer` implements the operation: `activation(dot(input, weights) + biases)`.
/// It is the most common and fundamental layer in many network architectures.
///
/// ### Example
/// ```dart
/// // A dense layer with 64 neurons and a ReLU activation function.
/// Layer layer1 = DenseLayer(64, activation: ReLU());
///
/// // A linear output layer with 10 neurons (no activation).
/// Layer outputLayer = DenseLayer(10);
/// ```
class DenseLayer extends Layer {
  @override
  final String name = 'dense';

  /// The number of neurons (or units) in the layer. This determines the
  /// dimensionality of the output space.
  final int outputSize;

  /// The activation function to apply after the linear transformation.
  /// If null, no activation is applied, and the layer is purely linear.
  final ActivationFunction? activation;

  DenseLayer(this.outputSize, {this.activation});

  late Tensor<Matrix> weights;
  late Tensor<Vector> biases;

  /// Provides the trainable weights and biases of this layer to the optimizer.
  @override
  List<Tensor> get parameters => [weights, biases];

  /// Initializes the `weights` and `biases` tensors.
  ///
  /// This method infers the `inputSize` from the input tensor's length and creates
  /// the weight matrix with the appropriate shape `[outputSize, inputSize]`. It uses
  /// **He initialization**, a best practice for layers that are followed by a
  /// ReLU activation, to promote stable training.
  @override
  void build(Tensor<dynamic> input) {
    if (input.value is! Vector) {
      throw Exception('DenseLayer requires a Vector input.');
    }
    int inputSize = (input.value as Vector).length;
    double stddev = sqrt(2.0 / inputSize);
    Random random = Random();

    List<List<double>> w = [];
    for (int i = 0; i < outputSize; i++) {
      Vector row = [];
      for (int j = 0; j < inputSize; j++) {
        row.add((sqrt(-2 * log(random.nextDouble())) * cos(2 * pi * random.nextDouble())) * stddev);
      }
      w.add(row);
    }

    weights = Tensor<Matrix>(w);
    biases = Tensor<Vector>(List<double>.filled(outputSize, 0.0));

    super.build(input);
  }

  /// Performs the forward pass for the dense layer.
  ///
  /// It first computes the linear transformation `z = Wx + b` and then
  /// applies the specified activation function, if one exists.
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