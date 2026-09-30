import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'activation_function.dart';

/// Applies the Exponential Linear Unit (ELU) activation function over a given [input] of type [Tensor<Vector>].
/// In contrast to the standard ReLU function, this function has a small negative value for negative inputs, which can help prevent the dying ReLU problem and speed up learning by pushing the mean activation closer to zero.
///
/// The function is defined as f(x) = x if x > 0, and f(x) = alpha * (e^x - 1) if x <= 0.
/// The value of [alpha] dictates the multiplier for negative inputs and is set to 1.0 by default.
/// The calculation is handled via the [eluVector] math operation.
///
/// Example:
/// ```dart
/// Tensor<Vector> v = Tensor<Vector>([-1.0, 0.0, 1.0]);
/// ELUVector elu = ELUVector(alpha: 1.0);
/// Tensor<Vector> out = elu(v);
/// ```
class ELUVector implements ActivationFunction<Vector> {
  double alpha;

  ELUVector({this.alpha = 1.0});

  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return eluVector(input, alpha);
  }
}

/// Applies the Exponential Linear Unit (ELU) activation function over a given [input] of type [Tensor<Matrix>].
/// In contrast to the standard ReLU function, this function has a small negative value for negative inputs, which can help prevent the dying ReLU problem and speed up learning by pushing the mean activation closer to zero.
///
/// The function is defined as f(x) = x if x > 0, and f(x) = alpha * (e^x - 1) if x <= 0.
/// The value of [alpha] dictates the multiplier for negative inputs and is set to 1.0 by default.
/// The calculation is handled via the [eluMatrix] math operation.
///
/// Example:
/// ```dart
/// Tensor<Matrix> m = Tensor<Matrix>([[-1.0, 2.0], [-3.0, 4.0]]);
/// ELUMatrix elu = ELUMatrix(alpha: 1.0);
/// Tensor<Matrix> out = elu(m);
/// ```
class ELUMatrix implements ActivationFunction<Matrix> {
  double alpha;

  ELUMatrix({this.alpha = 1.0});

  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return eluMatrix(input, alpha);
  }
}