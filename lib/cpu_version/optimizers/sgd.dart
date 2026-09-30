import 'optimizer.dart';
import '../../tensor/tensor.dart';

/// Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
/// Updates parameters in the opposite direction of the gradient scaled directly by [learningRate].
class SGD extends Optimizer {
  /// Creates an [SGD] optimizer for [parameters] with the specified [learningRate].
  SGD(super.parameters, super.learningRate);

  /// Performs a single optimization step, updating parameter values proportional to their gradients.
  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      for (int i = 0; i < param.data.length; i = i + 1) {
        param.data[i] = param.data[i] - learningRate * param.grad[i];
      }
    }
  }
}
