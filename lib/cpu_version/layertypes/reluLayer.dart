import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class ReLULayer extends Layer<Vector, Vector> {
  @override
  String name = 'relu_layer';

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    return relu(input);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}

class ReLULayerMatrix extends Layer<Matrix, Matrix> {
  @override
  String name = 'relu_layer_matrix';

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return reluMatrix(input);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}