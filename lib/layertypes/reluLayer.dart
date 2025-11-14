import '../autogradEngine/tensor.dart';
import 'layer.dart';

class ReLULayer extends Layer {
  @override
  String name = 'relu_layer';

  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    return relu(input as Tensor<Vector>);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}

class ReLULayerMatrix extends Layer {
  @override
  String name = 'relu_layer_matrix';

  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    return reluMatrix(input as Tensor<Matrix>);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}