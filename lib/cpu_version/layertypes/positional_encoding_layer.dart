import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import '../layertypes/layer.dart';

/// Injects sinusoidal positional encodings into an input sequence [Matrix] tensor.
/// Generates fixed sine and cosine frequency encodings up to [maxLength] across [dModel] feature dimensions to encode token positions,
/// adding them directly to the input tensor.
class PositionalEncoding extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'positional_encoding';

  /// Maximum sequence length supported for precomputed positional encodings.
  int maxLength;

  /// Dimensionality of the feature embeddings matching the input tensor column dimension.
  int dModel;

  /// Precomputed 2D matrix tensor of sinusoidal positional encodings of shape `[maxLength, dModel]`.
  late Tensor<Matrix> encodingMatrix;

  /// Creates a [PositionalEncoding] layer with [maxLength] and [dModel].
  PositionalEncoding(this.maxLength, this.dModel);

  /// Returns the trainable parameters. Since sinusoidal positional encodings are fixed and non-trainable, an empty list is returned.
  @override
  List<Tensor> get parameters => [];

  /// Precomputes the sinusoidal positional encoding matrix across all positions up to [maxLength].
  @override
  void build(Tensor<Matrix> input) {
    List<double> peValues = [];
    int totalElements = maxLength * dModel;

    for (int i = 0; i < totalElements; i = i + 1) {
      peValues.add(0.0);
    }

    for (int pos = 0; pos < maxLength; pos = pos + 1) {
      int posOffset = pos * dModel;
      for (int i = 0; i < dModel; i = i + 1) {
        double angle = pos / pow(10000, (2 * i) / dModel);
        if (i % 2 == 0) {
          peValues[posOffset + i] = sin(angle);
        } else {
          peValues[posOffset + i] = cos(angle);
        }
      }
    }

    encodingMatrix = Tensor<Matrix>(peValues);
    encodingMatrix.shape = [maxLength, dModel];

    super.build(input);
  }

  /// Slices the precomputed encoding matrix to match the input sequence length and adds it element-wise to [input] using [addMatrix].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    int sequenceLength = input.shape[0];
    int currentDModel = input.shape[1];

    // Slice out only the sequence length we need from the flat positional array
    List<double> applicableEncodings = [];
    int neededElements = sequenceLength * currentDModel;

    for (int i = 0; i < neededElements; i = i + 1) {
      applicableEncodings.add(encodingMatrix.data[i]);
    }

    Tensor<Matrix> positionalTensor = Tensor<Matrix>(applicableEncodings);
    positionalTensor.shape = [sequenceLength, currentDModel];

    return addMatrix(input, positionalTensor);
  }

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}
