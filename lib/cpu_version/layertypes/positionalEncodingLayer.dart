import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class PositionalEncoding extends Layer<Matrix, Matrix> {
  @override
  String name = 'positional_encoding';
  int maxLength;
  int dModel;

  late Tensor<Matrix> encodingMatrix;

  PositionalEncoding(this.maxLength, this.dModel);

  @override
  List<Tensor> get parameters => [];

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

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}