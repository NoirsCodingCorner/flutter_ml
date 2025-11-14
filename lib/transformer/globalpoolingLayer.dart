import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';

class GlobalAveragePooling1D extends Layer {
  @override
  String name = 'global_average_pooling_1d';
  late int sequenceLength;

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    sequenceLength = (input.value as Matrix).length;
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix inputMatrix = (input as Tensor<Matrix>).value;
    int numFeatures = inputMatrix[0].length;
    Vector sum = List<double>.filled(numFeatures, 0.0);

    for (int r = 0; r < inputMatrix.length; r++) {
      Vector row = inputMatrix[r];
      for (int i = 0; i < numFeatures; i++) {
        sum[i] += row[i];
      }
    }

    Vector outValue = [];
    for (int i = 0; i < numFeatures; i++) {
      outValue.add(sum[i] / sequenceLength);
    }

    Tensor<Vector> out = Tensor<Vector>(outValue);
    out.creator = Node([input], () {
      double distributed_grad = 1.0 / sequenceLength;
      for (int r = 0; r < sequenceLength; r++) {
        for (int c = 0; c < numFeatures; c++) {
          input.grad[r][c] += out.grad[c] * distributed_grad;
        }
      }
    }, opName: 'global_avg_pool_1d');

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