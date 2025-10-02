import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A Dropout layer for regularization.
///
/// Dropout is a technique used to prevent overfitting. During training, it
/// randomly sets a fraction of input units to 0 with a frequency of `rate`
//
/// at each step. This forces the network to learn more robust features.
///
/// **IMPORTANT:** This layer should only be active during training. During
/// evaluation or inference, it should be disabled or bypassed, allowing all
/// data to pass through unmodified.
///
/// This implementation uses "inverted dropout," where the outputs of the
/// non-dropped units are scaled up by `1 / (1 - rate)`. This ensures that
/// the expected output magnitude remains the same, and no changes are needed
/// at test time.
///
/// - **Input:** A `Tensor<Vector>` or `Tensor<Matrix>`.
/// - **Output:** A tensor of the same shape as the input.
///
/// ### Example
/// ```dart
/// SNetwork model = SNetwork([
///   DenseLayer(128, activation: ReLU()),
///   DropoutLayer(0.5), // Drops 50% of the inputs from the previous layer
///   DenseLayer(10),
/// ]);
/// ```
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
}

/// A Dropout layer for regularizing 2D Matrix data.
///
/// During training, it randomly sets a fraction of input units to 0 with a
/// frequency of `rate`. This is applied to each element of the matrix independently.
/// This layer should only be active during training.
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
}