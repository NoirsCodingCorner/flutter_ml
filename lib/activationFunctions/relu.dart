import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Rectified Linear Unit (ReLU) activation function.
///
/// ReLU is the most common activation function used in the hidden layers of
/// modern neural networks. It is computationally efficient and helps mitigate
/// the vanishing gradient problem.
///
/// The function is defined as `$f(x) = \max(0, x)$`. It outputs the input
/// directly if it's positive, and zero otherwise.
///
/// Its primary advantage is that its derivative is 1 for any positive input.
/// This allows the error signal (gradient) to flow backward through many
/// layers without shrinking, enabling the training of much deeper networks
/// compared to functions like `Sigmoid` or `Tanh`.
///
///
///
/// ### Example
/// ```dart
/// Layer hiddenLayer = DenseLayer(128, activation: ReLU());
/// ```
class ReLU implements ActivationFunction {
  /// Applies the ReLU function element-wise to the input tensor.
  ///
  /// This method calls the underlying `relu` operation, which handles both the
  /// forward calculation and the connection to the autograd graph for the
  /// backward pass.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return relu(input as Tensor<Vector>);
  }
}