

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

/// An activation function that applies the Rectified Linear Unit (ReLU) to a Vector.
///
/// ReLU is the most common activation function for hidden layers. It is defined
/// as $f(x) = \max(0, x)$, outputting the input if it is positive and zero otherwise.
///
/// This version is designed to work on 1D `Vector` inputs.
class ReLU implements ActivationFunction<Vector> {
  /// Applies the ReLU function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return relu(input);
  }
}

/// An activation function that applies the Rectified Linear Unit (ReLU) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the ReLU
/// function to each element independently. It's used after layers that process
/// batches of data, like `DenseLayerMatrix`.
class ReLUMatrix implements ActivationFunction<Matrix> {
  /// Applies the ReLU function element-wise to the input tensor.
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return reluMatrix(input);
  }
}