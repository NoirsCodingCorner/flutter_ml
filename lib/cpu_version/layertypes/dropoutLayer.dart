import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class DropoutLayer extends Layer<Vector, Vector> {
  @override
  String name = 'dropout';
  double rate;
  bool isTraining = true;

  DropoutLayer(this.rate);

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    return dropoutVectorMath(input, rate, isTraining);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}

class DropoutLayerMatrix extends Layer<Matrix, Matrix> {
  @override
  String name = 'dropout_matrix';
  double rate;
  bool isTraining = true;

  DropoutLayerMatrix(this.rate);

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return dropoutMatrixMath(input, rate, isTraining);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}
