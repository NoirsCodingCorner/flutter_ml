import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements the RMSprop optimization algorithm.
/// Maintains a moving average of squared gradients to normalize gradient magnitudes,
/// dividing the learning rate by the square root of recent gradient variances.
class RMSprop extends Optimizer {
  /// Exponential decay factor for the moving average of squared gradients.
  double beta;

  /// Small constant added to the denominator to prevent division by zero.
  double epsilon;

  /// Exponentially decaying average of past squared gradients mapped by tensor ID.
  late Map<String, List<double>> _s;

  /// Creates an [RMSprop] optimizer for [parameters] with the given [learningRate], decay factor [beta], and numerical stability constant [epsilon].
  RMSprop(
    List<Tensor<dynamic>> parameters, {
    required double learningRate,
    this.beta = 0.99,
    this.epsilon = 1e-8,
  }) : super(parameters, learningRate) {
    _s = {};
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      int size = param.data.length;
      List<double> sList = [];
      for (int i = 0; i < size; i = i + 1) {
        sList.add(0.0);
      }
      _s[param.id] = sList;
    }
  }

  /// Performs a single optimization step, updating moving squared gradients and adjusting parameter values.
  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> sList = _s[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];
        sList[i] = beta * sList[i] + (1.0 - beta) * (grad * grad);
        param.data[i] =
            param.data[i] - (learningRate * grad) / (sqrt(sList[i]) + epsilon);
      }
    }
  }
}
