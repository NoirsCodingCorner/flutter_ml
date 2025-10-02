import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Hyperbolic Tangent (Tanh) activation function.
///
/// It squashes any real-valued input into a range between -1 and 1. The function
/// is defined as `$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$`.
///
/// `Tanh` is the traditional activation function for the hidden states of recurrent
/// networks like `RNN`s and `LSTM`s. Its **zero-centered** output (ranging from
/// -1 to 1) often helps models learn more efficiently than the `Sigmoid` function.
///
/// However, like `Sigmoid`, it can still suffer from the vanishing gradient
/// problem in very deep networks.
///
///
///
/// ### Example
/// ```dart
/// // A simple RNN layer using Tanh as its activation.
/// Layer rnnLayer = RNN(32, activation: Tanh());
/// ```
class Tanh implements ActivationFunction {
  /// Applies the Tanh function element-wise to the input tensor.
  ///
  /// This method calls the underlying `vectorTanh` operation, which handles both
  /// the forward calculation and the connection to the autograd graph for the
  /// backward pass.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorTanh(input as Tensor<Vector>);
  }
}