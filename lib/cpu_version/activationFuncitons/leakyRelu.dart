import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

/// Applies leaky ReLu over a given [input] of type [Tensor<Vector>].
/// In contrast to the standard ReLU function, this function multiplies negative values with a small negative number [alpha] to prevent the dying gradient problem.
///
/// The value of [alpha] dictates the slope of the function for negative inputs and is set to `0.01` by default.
/// The calculation is handled via the [leakyReluVector] math operation.
///
/// Example:
/// ```dart
/// Tensor<Vector> v = Tensor<Vector>([-1.0, 0.0, 1.0]);
/// LeakyReLUVector leakyRelu = LeakyReLUVector(alpha: 0.02);
/// Tensor<Vector> out = leakyRelu(v);
/// ```
class LeakyReLUVector implements ActivationFunction<Vector> {
  double alpha;

  LeakyReLUVector({this.alpha = 0.01});

  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return leakyReluVector(input, alpha);
  }
}

/// Applies leaky ReLu over a given [input] of type [Tensor<Matrix>].
/// In contrast to the standard ReLU function, this function multiplies negative values with a small negative number [alpha] to prevent the dying gradient problem.
///
/// The value of [alpha] dictates the slope of the function for negative inputs and is set to `0.01` by default.
/// The calculation is handled via the [leakyReluMatrix] math operation.
///
/// Example:
/// ```dart
/// Tensor<Matrix> m = Tensor<Matrix>([[-1.0, 2.0], [-3.0, 4.0]]);
/// LeakyReLUMatrix leakyRelu = LeakyReLUMatrix(alpha: 0.01);
/// Tensor<Matrix> out = leakyRelu(m);
/// ```
class LeakyReLUMatrix implements ActivationFunction<Matrix> {
  double alpha;

  LeakyReLUMatrix({this.alpha = 0.01});

  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return leakyReluMatrix(input, alpha);
  }
}