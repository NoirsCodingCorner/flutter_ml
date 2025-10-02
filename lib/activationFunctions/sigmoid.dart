import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Sigmoid activation function.
///
/// It squashes any real-valued input into a range between 0 and 1. The function
/// is defined as `$f(x) = \frac{1}{1 + e^{-x}}$`.
///
/// Because of this property, it is the standard activation function for the
/// **output layer** in **binary classification** problems, where its output can be
/// interpreted as a probability (e.g., the probability that an email is spam).
///
/// While historically popular, it is rarely used in hidden layers of modern
/// networks as it can lead to the vanishing gradient problem. `ReLU` is
/// generally preferred for hidden layers.
///
///
///
/// ### Example
/// ```dart
/// // A binary classification output layer.
/// Layer outputLayer = DenseLayer(1, activation: Sigmoid());
/// ```
class Sigmoid implements ActivationFunction {
  /// Applies the Sigmoid function element-wise to the input tensor.
  ///
  /// This method calls the underlying `sigmoid` operation, which handles both the
  /// forward calculation and the connection to the autograd graph for the
  /// backward pass.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return sigmoid(input as Tensor<Vector>);
  }
}

/// An activation function that applies the Sigmoid function to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the Sigmoid
/// function to each element independently.
class SigmoidMatrix implements ActivationFunction {
  /// Applies the Sigmoid function element-wise to the input tensor.
  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return sigmoidMatrix(input as Tensor<Matrix>);
  }
}