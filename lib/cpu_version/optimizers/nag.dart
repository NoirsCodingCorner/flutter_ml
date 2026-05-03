import '../../tensor/tensor.dart';
import 'optimizer.dart';

class NAG extends Optimizer {
  double momentum;
  late Map<String, List<double>> _v;

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