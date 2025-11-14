import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../layertypes/layer.dart';
import 'embeddingLayer.dart';

class PositionalEncoding extends Layer {
  @override
  String name = 'positional_encoding';
  int maxLength;
  int dModel;

  late Tensor<Matrix> encodingMatrix;

  PositionalEncoding(this.maxLength, this.dModel);

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    Matrix pe = [];
    for (int i = 0; i < maxLength; i++) {
      Vector row = [];
      for (int j = 0; j < dModel; j++) {
        row.add(0.0);
      }
      pe.add(row);
    }

    for (int pos = 0; pos < maxLength; pos++) {
      for (int i = 0; i < dModel; i++) {
        double angle = pos / pow(10000, (2 * i) / dModel);
        if (i % 2 == 0) {
          pe[pos][i] = sin(angle);
        } else {
          pe[pos][i] = cos(angle);
        }
      }
    }
    encodingMatrix = Tensor<Matrix>(pe);
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;
    int sequenceLength = inputMatrix.value.length;

    Matrix applicableEncodings = [];
    for(int i=0; i < sequenceLength; i++){
      applicableEncodings.add(encodingMatrix.value[i]);
    }
    Tensor<Matrix> positionalTensor = Tensor<Matrix>(applicableEncodings);

    return addMatrix(inputMatrix, positionalTensor);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}