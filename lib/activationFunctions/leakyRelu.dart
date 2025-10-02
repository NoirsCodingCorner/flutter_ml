import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Leaky Rectified Linear Unit (Leaky ReLU) activation function.
///
/// This is a variant of the standard `ReLU` function. Instead of being zero for
/// negative inputs, `LeakyReLU` has a small negative slope (`alpha`), which
/// helps prevent the "Dying ReLU" problem and can lead to more robust training.
///
/// The function is defined as `$f(x) = x` if `$x > 0$`, and `$f(x) = \alpha \cdot x$`
/// if `$x \le 0$`. The `alpha` value is a small constant, typically 0.01.
///
///
///
/// ### Example
/// ```dart
/// // A hidden layer using LeakyReLU with a custom slope.
/// Layer hiddenLayer = DenseLayer(128, activation: LeakyReLU(alpha: 0.02));
/// ```
class LeakyReLU implements ActivationFunction {
  /// The small slope for negative inputs.
  final double alpha;

  LeakyReLU({this.alpha = 0.01});

  /// Applies the Leaky ReLU function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorLeakyReLU(input as Tensor<Vector>, alpha: alpha);
  }
}

/// Mathematical operation for element-wise Leaky ReLU on a vector.
Tensor<Vector> vectorLeakyReLU(Tensor<Vector> v, {double alpha = 0.01}) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] > 0 ? v.value[i] : alpha * v.value[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      v.grad[i] += out.grad[i] * (v.value[i] > 0 ? 1.0 : alpha);
    }
  }, opName: 'leaky_relu', cost: N);
  return out;
}