import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise over an input [Vector] tensor.
/// Computes `max(0, x)` on each element of the input vector.
class ReLULayer extends Layer<Vector, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'relu_layer';

  /// Returns the trainable parameters. Since ReLU contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Applies the ReLU activation function to the input vector on the CPU using [relu].
  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    return relu(input);
  }

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise over a 2D [Matrix] tensor.
/// Computes `max(0, x)` on each element across all rows and columns of the matrix.
class ReLULayerMatrix extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'relu_layer_matrix';

  /// Returns the trainable parameters. Since ReLU contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Applies the ReLU activation function to the input matrix on the CPU using [reluMatrix].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return reluMatrix(input);
  }

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}