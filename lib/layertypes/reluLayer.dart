import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// An activation layer that applies the Rectified Linear Unit (ReLU) function.
///
/// This version is designed to work on 1D `Vector` inputs.
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

/// An activation layer that applies ReLU element-wise to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, such as a
/// batch of samples or a sequence of vectors.
class ReLULayerMatrix extends Layer {
  @override
  String name = 'relu_layer_matrix';

  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    return reluMatrix(input as Tensor<Matrix>);
  }
}