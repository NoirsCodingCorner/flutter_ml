import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'activation_function.dart';

/// Applies the Sigmoid-weighted Linear Unit (SiLU / Swish) activation function to each element inside of a given [input] of type [Tensor<Vector>].
/// Defined mathematically as f(x) = x * sigmoid(x), this modern activation function often outperforms standard ReLU in deeper hidden layers.
///
/// The calculation is handled via the [swishVector] math operation.
///
/// Example:
/// ```dart
/// Tensor<Vector> v = Tensor<Vector>([1.0, 2.0, 3.0]);
/// SwishVector swish = SwishVector();
/// Tensor<Vector> out = swish(v);
/// ```
class SwishVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return swishVector(input);
  }
}

/// Applies the Sigmoid-weighted Linear Unit (SiLU / Swish) activation function to each element inside of a given [input] of type [Tensor<Matrix>].
/// Defined mathematically as f(x) = x * sigmoid(x), this modern activation function often outperforms standard ReLU in deeper hidden layers.
///
/// The calculation is handled via the [swishMatrix] math operation.
///
/// Example:
/// ```dart
/// Tensor<Matrix> m = Tensor<Matrix>([[1.0, 2.0], [3.0, 4.0]]);
/// SwishMatrix swish = SwishMatrix();
/// Tensor<Matrix> out = swish(m);
/// ```
class SwishMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return swishMatrix(input);
  }
}
