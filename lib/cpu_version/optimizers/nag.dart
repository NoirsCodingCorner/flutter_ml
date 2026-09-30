import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements Nesterov Accelerated Gradient (NAG) optimization.
/// Incorporates momentum by computing the parameter update step using a look-ahead velocity term.
class NAG extends Optimizer {
  /// The momentum factor determining the contribution of past update velocities.
  double momentum;

  /// Velocity vectors accumulated over previous steps mapped by tensor ID.
  late Map<String, List<double>> _v;

  /// Creates a [NAG] optimizer for [parameters] with the given [learningRate] and [momentum] factor.
  NAG(
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

  /// Performs a single optimization step using the Nesterov momentum update rule.
  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> vList = _v[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];
        double vOld = vList[i];
        double vNew = momentum * vOld + grad;
        vList[i] = vNew;

        // NAG update rule
        param.data[i] = param.data[i] - learningRate * (grad + momentum * vNew);
      }
    }
  }
}