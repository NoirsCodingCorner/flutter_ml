import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// An activation function that applies the Rectified Linear Unit (ReLU) to a Vector.
///
/// ReLU is the most common activation function for hidden layers. It is defined
/// as `$f(x) = \max(0, x)$`, outputting the input if it is positive and zero otherwise.
///
/// This version is designed to work on 1D `Vector` inputs.
class ReLU implements ActivationFunction {
  /// Applies the ReLU function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return relu(input as Tensor<Vector>);
  }
}

/// An activation function that applies the Rectified Linear Unit (ReLU) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the ReLU
/// function to each element independently. It's used after layers that process
/// batches of data, like `DenseLayerMatrix`.
class ReLUMatrix implements ActivationFunction {
  /// Applies the ReLU function element-wise to the input tensor.
  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return reluMatrix(input as Tensor<Matrix>);
  }
}