
import '../autogradEngine/tensor.dart';

/// The abstract base class (or interface) for all activation functions.
///
/// The purpose of this class is to define a common contract that all
/// activation functions, such as `ReLU`, `Sigmoid`, or `Tanh`, must follow.
///
/// By having this common interface, layers like `DenseLayer` can be written to
/// work with any activation function, making the framework modular and easy to
/// extend with new, custom activations.
///
/// ### Example
/// ```dart
/// // Both layers accept an object that implements ActivationFunction.
/// Layer denseWithReLU = DenseLayer(64, activation: ReLU());
/// Layer denseWithTanh = DenseLayer(64, activation: Tanh());
/// ```
abstract class ActivationFunction {
  /// The sole method that subclasses must implement.
  ///
  /// It takes an input tensor, applies the specific activation function's logic,
  /// and returns the resulting output tensor. The underlying mathematical
  /// function is responsible for connecting the operation to the autograd graph.
  Tensor<dynamic> call(Tensor<dynamic> input);
}
