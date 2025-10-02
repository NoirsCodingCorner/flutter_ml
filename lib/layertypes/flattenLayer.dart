import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A utility layer that flattens a multi-dimensional tensor into a 1D vector.
///
/// A `FlattenLayer` has no trainable parameters. Its primary role is to act as
/// a bridge between layers that output multi-dimensional data (like `ConvLSTMLayer`
/// or `Conv2D`) and layers that expect a 1D vector input (like `DenseLayer`).
///
/// ### Example
/// ```dart
/// SNetwork model = SNetwork([
///   ConvLSTMLayer(8, 3), // Outputs a 2D feature map
///   FlattenLayer(),      // Flattens the map into a vector
///   DenseLayer(1),       // Processes the vector
/// ]);
/// ```
class FlattenLayer extends Layer {
  @override
  String name = 'flatten';
  late int inputRows, inputCols;

  /// Returns an empty list as this layer has no trainable parameters.
  @override
  List<Tensor> get parameters => [];

  /// Records the input dimensions, which are needed for the backward pass.
  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    inputRows = inputMatrix.length;
    inputCols = inputMatrix[0].length;
    super.build(input);
  }

  /// Unrolls the input matrix into a single, long vector.
  ///
  /// The backward pass correctly reshapes the incoming 1D gradient back into
  /// a 2D matrix to be passed to the previous layer.
  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix inputMatrix = (input as Tensor<Matrix>).value;
    Vector flatList = [];
    for (Vector row in inputMatrix) {
      flatList.addAll(row);
    }
    Tensor<Vector> out = Tensor<Vector>(flatList);

    out.creator = Node([input], () {
      int index = 0;
      for (int i = 0; i < inputRows; i++) {
        for (int j = 0; j < inputCols; j++) {
          input.grad[i][j] += out.grad[index];
          index++;
        }
      }
    }, opName: 'flatten', cost: 0);
    return out;
  }
}