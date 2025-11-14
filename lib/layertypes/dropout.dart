import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

class DropoutLayer extends Layer {
  @override
  String name = 'dropout';
  double rate;
  bool isTraining = true;

  DropoutLayer(this.rate);

  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    if (isTraining == false || rate == 0) {
      return input as Tensor<Vector>;
    }

    double scale = 1.0 / (1.0 - rate);
    Random random = Random();
    Vector inputValue = input.value as Vector;
    Vector outputValue = [];
    List<bool> mask = [];

    for (int i = 0; i < inputValue.length; i++) {
      if (random.nextDouble() < rate) {
        outputValue.add(0.0);
        mask.add(false);
      } else {
        outputValue.add(inputValue[i] * scale);
        mask.add(true);
      }
    }
    Tensor<Vector> out = Tensor<Vector>(outputValue);
    out.creator = Node([input], () {
      for (int i = 0; i < inputValue.length; i++) {
        if (mask[i]) {
          input.grad[i] += out.grad[i] * scale;
        }
      }
    }, opName: 'dropout', cost: inputValue.length);
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

class DropoutLayerMatrix extends Layer {
  @override
  String name = 'dropout_matrix';
  double rate;
  bool isTraining = true;

  DropoutLayerMatrix(this.rate);

  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    if (isTraining == false || rate == 0) {
      return input as Tensor<Matrix>;
    }

    double scale = 1.0 / (1.0 - rate);
    Random random = Random();
    Matrix inputMatrix = input.value as Matrix;
    Matrix outputValue = [];
    List<List<bool>> mask = [];

    for (int r = 0; r < inputMatrix.length; r++) {
      Vector row = [];
      List<bool> maskRow = [];
      for (int c = 0; c < inputMatrix[0].length; c++) {
        if (random.nextDouble() < rate) {
          row.add(0.0);
          maskRow.add(false);
        } else {
          row.add(inputMatrix[r][c] * scale);
          maskRow.add(true);
        }
      }
      outputValue.add(row);
      mask.add(maskRow);
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputValue);
    out.creator = Node([input], () {
      for (int r = 0; r < inputMatrix.length; r++) {
        for (int c = 0; c < inputMatrix[0].length; c++) {
          if (mask[r][c]) {
            input.grad[r][c] += out.grad[r][c] * scale;
          }
        }
      }
    }, opName: 'dropout_matrix', cost: inputMatrix.length * inputMatrix[0].length);
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