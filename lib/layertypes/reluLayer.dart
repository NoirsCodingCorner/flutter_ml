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
}