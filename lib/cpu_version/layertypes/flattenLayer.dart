import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class FlattenLayer extends Layer<Matrix, Vector> {
  @override
  String name = 'flatten';
  late int inputRows;
  late int inputCols;

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

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

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}