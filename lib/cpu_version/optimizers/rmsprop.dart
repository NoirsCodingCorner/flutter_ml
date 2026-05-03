import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

class RMSprop extends Optimizer {
  double beta;
  double epsilon;
  late Map<String, List<double>> _s;

  RMSprop(
      List<Tensor<dynamic>> parameters, {
        required double learningRate,
        this.beta = 0.99,
        this.epsilon = 1e-8,
      }) : super(parameters, learningRate: learningRate) {
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

  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> sList = _s[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double grad = param.grad[i];
        sList[i] = beta * sList[i] + (1.0 - beta) * (grad * grad);
        param.data[i] = param.data[i] - (learningRate * grad) / (sqrt(sList[i]) + epsilon);
      }
    }
  }
}