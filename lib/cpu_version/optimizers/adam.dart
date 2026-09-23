import 'dart:math';

import 'optimizer.dart';

import '../../tensor/tensor.dart';


class Adam extends Optimizer {
  double beta1;
  double beta2;
  double epsilon;
  int _t = 0;

  late Map<String, List<double>> _m;
  late Map<String, List<double>> _v;

  Adam(
      List<Tensor<dynamic>> parameters, {
        required double learningRate,
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-8,
      }) : super(parameters, learningRate: learningRate) {
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

      // Using the unique tensor ID as the key
      _m[param.id] = mList;
      _v[param.id] = vList;
    }
  }

  @override
  void step() {
    _t = _t + 1;
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> mList = _m[param.id]!;
      List<double> vList = _v[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        mList[i] = beta1 * mList[i] + (1.0 - beta1) * param.grad[i];
        vList[i] = beta2 * vList[i] + (1.0 - beta2) * (param.grad[i] * param.grad[i]);

        double mHat = mList[i] / (1.0 - pow(beta1, _t));
        double vHat = vList[i] / (1.0 - pow(beta2, _t));

        param.data[i] = param.data[i] - learningRate * mHat / (sqrt(vHat) + epsilon);
      }
    }
  }
}