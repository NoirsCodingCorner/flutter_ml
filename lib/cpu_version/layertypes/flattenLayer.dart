import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

/// Flattens a 2D [Matrix] tensor into a contiguous 1D [Vector] tensor.
/// Preserves the underlying flat data elements while resetting dimensionality and attaching an autograd [Node] to propagate gradients backward.
class FlattenLayer extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'flatten';

  /// Cached row dimension of the input matrix.
  late int inputRows;

  /// Cached column dimension of the input matrix.
  late int inputCols;

  /// Returns the trainable parameters. Since flattening contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Caches the spatial row and column dimensions of the input matrix and marks the layer as built.
  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    inputRows = inputMatrix.length;
    if (inputRows > 0) {
      inputCols = inputMatrix[0].length;
    } else {
      inputCols = 0;
    }
    super.build(input);
  }

  /// Executes the flattening operation on the CPU by copying the flat data buffer and registering an autograd node.
  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    // Because the Tensor class natively flattens all data internally,
    // we can skip complex matrix loops and just copy the flat 1D data.
    Vector flatList = [];
    for (int i = 0; i < input.data.length; i = i + 1) {
      flatList.add(input.data[i]);
    }

    Tensor<Vector> out = Tensor<Vector>(flatList);

    out.creator = Node(
      [input],
          () {
        for (int i = 0; i < input.data.length; i = i + 1) {
          input.grad[i] = input.grad[i] + out.grad[i];
        }
      },
      opName: 'flatten',
      cost: 0,
    );
    return out;
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