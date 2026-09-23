import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

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
class LeakyReLUVector implements ActivationFunction<Vector> {
  double alpha;

  LeakyReLUVector({this.alpha = 0.01});

  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return leakyReluVector(input, alpha);
  }
}

class LeakyReLUMatrix implements ActivationFunction<Matrix> {
  double alpha;

  LeakyReLUMatrix({this.alpha = 0.01});

  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return leakyReluMatrix(input, alpha);
  }
}