import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'activation_function.dart';

/// Applies the mish activation function (similar to swish) to each element inside of a given [input] of type [Tensor<Vector>].
/// Mish mirrors Relu with a continuous curve.
/// The calculation is handled via the [mishVector] math operation.
///
/// Example:
/// ```dart
/// Tensor<Vector> v = Tensor<Vector>([1.0, 2.0, 3.0]);
/// MishVector mish = MishVector();
/// Tensor<Vector> out = mish(v);
/// ```
class MishVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return mishVector(input);
  }
}

/// Applies the mish activation function (similar to swish) to each element inside of a given [input] of type [Tensor<Matrix>].
/// Mish mirrors Relu with a continuous curve.
/// The calculation is handled via the [mishMatrix] math operation.
///
/// Example:
/// ```dart
/// Tensor<Matrix> m = Tensor<Matrix>([[1.0, 2.0], [3.0, 4.0]]);
/// MishMatrix mish = MishMatrix();
/// Tensor<Matrix> out = mish(m);
/// ```
class MishMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return mishMatrix(input);
  }
}