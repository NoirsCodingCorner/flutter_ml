import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

/// The Sigmoid-weighted Linear Unit (SiLU) activation function for Vectors.
class SwishVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return swishVector(input);
  }
}

/// The Sigmoid-weighted Linear Unit (SiLU) activation function for Matrices.
class SwishMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return swishMatrix(input);
  }
}