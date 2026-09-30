import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements the Adagrad optimization algorithm.
/// Adapts the learning rate individually for each parameter based on the accumulated history of squared gradients.
class Adagrad extends Optimizer {
  /// Small constant added to the denominator to prevent division by zero.
  double epsilon;

  /// Accumulated sum of squared gradients mapped by tensor ID.
  late Map<String, List<double>> _gSquaredSum;

  /// Creates an [Adagrad] optimizer for [parameters] with the specified [learningRate] and [epsilon].
  Adagrad(
    List<Tensor<dynamic>> parameters, {
    required double learningRate,
    this.epsilon = 1e-8,
  }) : super(parameters, learningRate: learningRate) {
    _gSquaredSum = {};
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      int size = param.data.length;
      List<double> stateList = [];
      for (int i = 0; i < size; i = i + 1) {
        stateList.add(0.0);
      }
      _gSquaredSum[param.id] = stateList;
    }
  }

  /// Performs a single optimization step, accumulating squared gradients and updating parameter values.
  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> gSum = _gSquaredSum[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double gradient = param.grad[i];
        gSum[i] = gSum[i] + (gradient * gradient);
        param.data[i] = param.data[i] -
            (learningRate * gradient) / (sqrt(gSum[i]) + epsilon);
      }
    }
  }
}
