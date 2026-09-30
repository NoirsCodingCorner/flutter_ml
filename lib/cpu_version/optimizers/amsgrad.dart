import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

/// Implements the AMSGrad optimization algorithm, a variant of Adam.
/// Maintains the maximum of past squared gradient averages in [_vHat] to prevent learning rate scaling from exploding,
/// ensuring non-increasing effective step sizes for improved convergence.
class AMSGrad extends Optimizer {
  /// Exponential decay rate for the first moment estimates.
  double beta1;

  /// Exponential decay rate for the second moment estimates.
  double beta2;

  /// Small constant added to the denominator to prevent division by zero.
  double epsilon;

  /// Timestep counter tracking the number of optimization steps taken.
  int _t = 0;

  /// First moment vector estimates mapped by tensor ID.
  late Map<String, List<double>> _m;

  /// Second moment vector estimates mapped by tensor ID.
  late Map<String, List<double>> _v;

  /// Maximum historical second moment vector estimates mapped by tensor ID.
  late Map<String, List<double>> _vHat;

  /// Creates an [AMSGrad] optimizer for [parameters] with [learningRate], moment decay rates [beta1] and [beta2], and numerical stability constant [epsilon].
  AMSGrad(
    List<Tensor<dynamic>> parameters, {
    required double learningRate,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  }) : super(parameters, learningRate) {
    _m = {};
    _v = {};
    _vHat = {};
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      int size = param.data.length;

      List<double> mList = [];
      List<double> vList = [];
      List<double> vHatList = [];
      for (int i = 0; i < size; i = i + 1) {
        mList.add(0.0);
        vList.add(0.0);
        vHatList.add(0.0);
      }

      _m[param.id] = mList;
      _v[param.id] = vList;
      _vHat[param.id] = vHatList;
    }
  }

  /// Performs a single optimization step, updating first moments, tracking maximum second moments in [_vHat], and updating parameters.
  @override
  void step() {
    _t = _t + 1;
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> mList = _m[param.id]!;
      List<double> vList = _v[param.id]!;
      List<double> vHatList = _vHat[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];

        mList[i] = beta1 * mList[i] + (1.0 - beta1) * grad;
        vList[i] = beta2 * vList[i] + (1.0 - beta2) * (grad * grad);

        // Use the maximum of past squared gradients
        if (vList[i] > vHatList[i]) {
          vHatList[i] = vList[i];
        }

        double mHat = mList[i] / (1.0 - pow(beta1, _t));
        param.data[i] =
            param.data[i] - learningRate * mHat / (sqrt(vHatList[i]) + epsilon);
      }
    }
  }
}
