import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'layer.dart';

/// Applies Dropout regularization over an input [Vector] tensor.
/// During training, randomly zeroes out elements of the input tensor with probability [rate]
/// and scales the remaining elements by `1.0 / (1.0 - rate)` to maintain expected activation magnitude.
/// During evaluation ([isTraining] = false), the input tensor is passed through unchanged.
class DropoutLayer extends Layer<Vector, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'dropout';

  /// The probability of an element being dropped to zero.
  double rate;

  /// Bool flag indicating whether the layer is actively dropping units during training or acting as an identity pass during inference.
  bool isTraining = true;

  /// Creates a [DropoutLayer] with the specified dropout [rate].
  DropoutLayer(this.rate);

  /// Returns the trainable parameters. Since dropout contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Executes the vector dropout operation on the CPU using [dropoutVectorMath].
  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    return dropoutVectorMath(input, rate, isTraining);
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

/// Applies Dropout regularization over a 2D [Matrix] tensor for batched or spatial representations.
/// During training, randomly zeroes out elements of the input matrix with probability [rate]
/// and scales the remaining elements by `1.0 / (1.0 - rate)` to maintain expected activation magnitude.
/// During evaluation ([isTraining] = false), the input tensor is passed through unchanged.
class DropoutLayerMatrix extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'dropout_matrix';

  /// The probability of an element being dropped to zero.
  double rate;

  /// Bool flag indicating whether the layer is actively dropping units during training or acting as an identity pass during inference.
  bool isTraining = true;

  /// Creates a [DropoutLayerMatrix] with the specified dropout [rate].
  DropoutLayerMatrix(this.rate);

  /// Returns the trainable parameters. Since dropout contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Executes the matrix dropout operation on the CPU using [dropoutMatrixMath].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return dropoutMatrixMath(input, rate, isTraining);
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
