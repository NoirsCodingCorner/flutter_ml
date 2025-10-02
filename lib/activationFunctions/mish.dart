import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Mish activation function.
///
/// Mish is a modern, self-gated, and smooth non-monotonic activation function
/// that has achieved state-of-the-art results on a number of computer vision
/// and NLP benchmarks, often outperforming `ReLU` and `Swish`.
///
/// The function is defined as `$f(x) = x \cdot \tanh(\text{softplus}(x))$`.
///
///
///
/// ### Example
/// ```dart
/// Layer hiddenLayer = DenseLayer(128, activation: Mish());
/// ```
class Mish implements ActivationFunction {
  /// Applies the Mish function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorMish(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the Mish function on a vector.
///
/// Built by composing other autograd operations. The backward pass is handled automatically.
Tensor<Vector> vectorMish(Tensor<Vector> x) {
  return elementWiseMultiply(x, vectorTanh(softplus(x)));
}