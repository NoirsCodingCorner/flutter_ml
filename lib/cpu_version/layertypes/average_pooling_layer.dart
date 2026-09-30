import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'layer.dart';

/// Applies a 2D average pooling operation over an input [Matrix] tensor.
/// Downsamples the input representation by taking the average value over pooling windows defined by [poolSize] and [stride].
class AveragePooling2DLayer extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'average_pooling_2d';

  /// The size of the sliding pooling window.
  int poolSize;

  /// The step size of the pooling window across the input matrix.
  int stride;

  /// Cached height dimension of the input [Tensor].
  late int inputHeight;

  /// Cached width dimension of the input [Tensor].
  late int inputWidth;

  /// Creates an [AveragePooling2DLayer] with the given [poolSize] and [stride].
  AveragePooling2DLayer({this.poolSize = 2, this.stride = 2});

  /// Returns the trainable parameters. Since average pooling contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Extracts the spatial dimensions from the input [Tensor] and registers the layer as built.
  @override
  void build(Tensor<Matrix> input) {
    inputHeight = input.shape[0];
    inputWidth = input.shape[1];
    super.build(input);
  }

  /// Performs the 2D average pooling operation on the CPU using [avgPool2d].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return avgPool2d(input, poolSize, stride);
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

/// Applies a global average pooling operation over an input [Matrix] tensor.
/// Computes the mean along the spatial dimensions to reduce a 2D [Matrix] to a 1D [Vector].
class GlobalAveragePoolingLayer extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'global_avg_pool';

  /// Returns the trainable parameters. Since global average pooling contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Executes global average pooling on the input [Matrix] using [globalAveragePooling] and returns a [Vector] [Tensor].
  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    return globalAveragePooling(input);
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
