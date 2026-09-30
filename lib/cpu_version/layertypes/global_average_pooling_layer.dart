import '../../tensor/tensor.dart';
import '../../tensor/type_aliases.dart';
import '../layertypes/layer.dart';

/// Applies a 1D global average pooling operation over an input [Matrix] sequence of shape `[sequenceLength, numFeatures]`.
/// Computes the mean across all sequence timesteps for each feature dimension, reducing the 2D [Matrix] to a 1D [Vector] of size `[numFeatures]`.
/// Registers an autograd [Node] to evenly distribute incoming gradients back across all sequence steps.
class GlobalAveragePooling1D extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'global_average_pooling_1d';

  /// Cached number of timesteps in the incoming sequence.
  late int sequenceLength;

  /// Cached number of features per timestep.
  late int numFeatures;

  /// Returns the trainable parameters. Since global average pooling contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor> get parameters => [];

  /// Extracts the sequence length and feature count from the input [Tensor] and registers the layer as built.
  @override
  void build(Tensor<Matrix> input) {
    sequenceLength = input.shape[0];
    numFeatures = input.shape[1];
    super.build(input);
  }

  /// Executes 1D global average pooling on the CPU by averaging columns across rows and attaches a backward [Node] for gradient propagation.
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
          input.grad[offset + c] =
              input.grad[offset + c] + out.grad[c] * distributedGrad;
        }
      }
    }, opName: 'global_avg_pool_1d');

    return out;
  }

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() => {};

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}
