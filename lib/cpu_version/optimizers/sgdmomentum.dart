import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements the Stochastic Gradient Descent with Momentum optimization algorithm.
/// Accelerates gradient vectors in the relevant direction by accumulating past velocity updates,
/// dampening oscillations across optimization steps.
class Momentum extends Optimizer {
  /// The momentum factor determining the exponential decay of past update velocities.
  double momentum;

  /// Accumulated velocity vectors mapped by tensor ID.
  late Map<String, List<double>> _v;

  /// Creates a [Momentum] optimizer for [parameters] with [learningRate] and [momentum] factor.
  Momentum(
    List<Tensor<dynamic>> parameters, {
    required double learningRate,
    this.momentum = 0.9,
  }) : super(parameters, learningRate: learningRate) {
    _v = {};
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      int size = param.data.length;
      List<double> vList = [];
      for (int i = 0; i < size; i = i + 1) {
        vList.add(0.0);
      }
      _v[param.id] = vList;
    }
  }

  /// Performs a single optimization step, accumulating velocity vectors and updating parameter values.
  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> vList = _v[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];
        vList[i] = momentum * vList[i] + learningRate * grad;
        param.data[i] = param.data[i] - vList[i];
      }
    }
  }
}
