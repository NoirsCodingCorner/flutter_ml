import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

/// An activation function that applies the Exponential Linear Unit (ELU) to a Vector.
///
/// ELU is an alternative to `ReLU` that has a small negative value for negative
/// inputs, which can help prevent the "Dying ReLU" problem and speed up learning.
///
/// The function is defined as `$f(x) = x` if `$x > 0$`, and `$f(x) = \alpha(e^x - 1)$` if `$x \le 0$`.
class ELUVector implements ActivationFunction<Vector> {
  double alpha;

  ELUVector({this.alpha = 1.0});

  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return eluVector(input, alpha);
  }
}
/// An activation function that applies the Exponential Linear Unit (ELU) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the ELU
/// function to each element independently.
class ELUMatrix implements ActivationFunction<Matrix> {
  double alpha;

  ELUMatrix({this.alpha = 1.0});

  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return eluMatrix(input, alpha);
  }
}