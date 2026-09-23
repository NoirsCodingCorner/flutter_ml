
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';

/// An activation function that applies Softmax to a Vector.
///
/// Softmax converts a vector of real numbers (logits) into a probability
/// distribution where all elements sum to 1. It is the standard activation
/// for the output layer in multi-class classification problems.

/// An activation function that applies Softmax to a Vector.
class SoftmaxVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return softmaxVector(input);
  }
}

/// An activation function that applies Softmax to each row of a Matrix.
class SoftmaxMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return softmaxMatrix(input);
  }
}