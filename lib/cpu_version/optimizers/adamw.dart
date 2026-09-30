import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements the AdamW optimization algorithm with decoupled weight decay.
/// Modifies the standard Adam update by applying weight decay directly to the parameters rather than incorporating it into the gradient moments.
class AdamW extends Optimizer {
  /// Exponential decay rate for the first moment estimates.
  double beta1;

  /// Exponential decay rate for the second moment estimates.
  double beta2;

  /// Small constant added to the denominator to prevent division by zero.
  double epsilon;

  /// Weight decay coefficient applied independently of gradient moment updates.
  double weightDecay;

  /// Timestep counter tracking the number of optimization steps taken.
  int _t = 0;

  /// First moment vector estimates mapped by tensor ID.
  late Map<String, List<double>> _m;

  /// Second moment vector estimates mapped by tensor ID.
  late Map<String, List<double>> _v;

  /// Creates an [AdamW] optimizer for [parameters] with [learningRate], decay rates [beta1] and [beta2], [epsilon], and decoupled [weightDecay].
  AdamW(
    List<Tensor<dynamic>> parameters, {
    required double learningRate,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    this.weightDecay = 0.01,
  }) : super(parameters, learningRate) {
    _m = {};
    _v = {};
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      int size = param.data.length;

      List<double> mList = [];
      List<double> vList = [];
      for (int i = 0; i < size; i = i + 1) {
        mList.add(0.0);
        vList.add(0.0);
      }

      _m[param.id] = mList;
      _v[param.id] = vList;
    }
  }

  /// Performs a single optimization step, updating moment estimates, applying decoupled weight decay directly to parameter values, and adjusting parameters.
  @override
  void step() {
    _t = _t + 1;
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> mList = _m[param.id]!;
      List<double> vList = _v[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];

        // Update moments
        mList[i] = beta1 * mList[i] + (1.0 - beta1) * grad;
        vList[i] = beta2 * vList[i] + (1.0 - beta2) * (grad * grad);

        // Bias correction
        double mHat = mList[i] / (1.0 - pow(beta1, _t));
        double vHat = vList[i] / (1.0 - pow(beta2, _t));

        // Weight decay (decoupled from gradient update)
        param.data[i] =
            param.data[i] - (learningRate * weightDecay * param.data[i]);

        // Gradient update
        param.data[i] =
            param.data[i] - (learningRate * mHat) / (sqrt(vHat) + epsilon);
      }
    }
  }
}
