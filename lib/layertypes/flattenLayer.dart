import '../autogradEngine/tensor.dart';
import 'layer.dart';

class FlattenLayer extends Layer {
  @override
  String name = 'flatten';
  late int inputRows, inputCols;

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    inputRows = inputMatrix.length;
    inputCols = inputMatrix[0].length;
    super.build(input);
  }

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

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}