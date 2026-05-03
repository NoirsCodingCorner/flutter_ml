
import 'optimizer.dart';
import '../../tensor/tensor.dart';

/// Implements the Stochastic Gradient Descent (SGD) optimizer.
class SGD extends Optimizer {
  SGD(super.parameters, {required super.learningRate});

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