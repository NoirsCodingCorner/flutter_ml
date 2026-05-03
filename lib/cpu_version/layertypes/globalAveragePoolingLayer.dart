import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class GlobalAveragePooling1D extends Layer<Matrix, Vector> {
  @override
  String name = 'global_average_pooling_1d';
  late int sequenceLength;
  late int numFeatures;

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<Matrix> input) {
    sequenceLength = input.shape[0];
    numFeatures = input.shape[1];
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    List<double> sum = List<double>.filled(numFeatures, 0.0);

    for (int r = 0; r < sequenceLength; r = r + 1) {
      int offset = r * numFeatures;
      for (int c = 0; c < numFeatures; c = c + 1) {
        sum[c] = sum[c] + input.data[offset + c];
      }
    }

    List<double> outValue = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      outValue.add(sum[i] / sequenceLength);
    }

    Tensor<Vector> out = Tensor<Vector>(outValue);
    out.shape = [numFeatures];

    out.creator = Node([input], () {
      double distributedGrad = 1.0 / sequenceLength;
      for (int r = 0; r < sequenceLength; r = r + 1) {
        int offset = r * numFeatures;
        for (int c = 0; c < numFeatures; c = c + 1) {
          input.grad[offset + c] = input.grad[offset + c] + out.grad[c] * distributedGrad;
        }
      }
    }, opName: 'global_avg_pool_1d');

    return out;
  }

  @override
  Map<String, dynamic> getWeights() => {};

  @override
  void setWeights(Map<String, dynamic> weights) {}
}