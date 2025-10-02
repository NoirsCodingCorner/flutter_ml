import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';

/// A Global Average Pooling layer for 1D data.
///
/// This layer takes a matrix of shape `[sequence_length, features]` and
/// computes the average across the sequence dimension, resulting in a single
/// vector of shape `[features]`.
///
/// It is a simple way to aggregate sequence information for a final classification.
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

    for (Vector row in inputMatrix) {
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
      // The backward pass distributes the gradient evenly to all timesteps
      double distributed_grad = 1.0 / sequenceLength;
      for (int r = 0; r < sequenceLength; r++) {
        for (int c = 0; c < numFeatures; c++) {
          input.grad[r][c] += out.grad[c] * distributed_grad;
        }
      }
    }, opName: 'global_avg_pool_1d');

    return out;
  }
}