import 'dart:math';
import '../../tensor/tensor.dart';
import 'optimizer.dart';

class Adagrad extends Optimizer {
  double epsilon;
  late Map<String, List<double>> _gSquaredSum;

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

  @override
  void step() {
    for (int p = 0; p < parameters.length; p = p + 1) {
      Tensor<dynamic> param = parameters[p];
      List<double> gSum = _gSquaredSum[param.id]!;

      for (int i = 0; i < param.data.length; i = i + 1) {
        double gradient = param.grad[i];
        gSum[i] = gSum[i] + (gradient * gradient);
        param.data[i] = param.data[i] - (learningRate * gradient) / (sqrt(gSum[i]) + epsilon);
      }
    }
  }
}