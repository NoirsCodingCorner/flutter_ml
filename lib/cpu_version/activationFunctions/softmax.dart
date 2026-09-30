import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'activation_function.dart';

/// Applies the softmax function over a given [input] of type [Tensor<Vector>].
/// Creates a probability distribution over all values summing up to 1.
/// The calculation is handled via the [softmaxVector] math operation.
///
/// Example:
/// ```dart
/// Tensor<Vector> v = Tensor<Vector>([1.0, 2.0, 3.0]);
/// SoftmaxVector softmax = SoftmaxVector();
/// Tensor<Vector> out = softmax(v);
/// ```
class SoftmaxVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return softmaxVector(input);
  }
}

/// Applies the softmax function over a given [input] of type [Tensor<Matrix>].
/// Creates a probability distribution over all values of each row summing up to 1.
/// The calculation is handled via the [softmaxMatrix] math operation.
///
/// Example:
/// ```dart
/// Tensor<Matrix> m = Tensor<Matrix>([[1.0, 2.0], [3.0, 4.0]]);
/// SoftmaxMatrix softmax = SoftmaxMatrix();
/// Tensor<Matrix> out = softmax(m);
/// ```
class SoftmaxMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return softmaxMatrix(input);
  }
}