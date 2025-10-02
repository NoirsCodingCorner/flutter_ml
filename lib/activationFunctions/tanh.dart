import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';


/// An activation function that applies the Hyperbolic Tangent (Tanh) to a Vector.
///
/// It squashes any real-valued input into a range between -1 and 1. `Tanh` is
/// the traditional activation function for the hidden states of recurrent
/// networks like `RNN`s and `LSTM`s.
///
/// This version is designed to work on 1D `Vector` inputs.
class Tanh implements ActivationFunction {
  /// Applies the Tanh function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorTanh(input as Tensor<Vector>);
  }
}

/// An activation function that applies the Hyperbolic Tangent (Tanh) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the Tanh
/// function to each element independently.
class TanhMatrix implements ActivationFunction {
  /// Applies the Tanh function element-wise to the input tensor.
  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return tanhMatrix(input as Tensor<Matrix>);
  }
}

/// Mathematical operation for the Tanh function on a matrix.
Tensor<Matrix> tanhMatrix(Tensor<Matrix> m) {
  double _tanh(double x) {
    double e2x = exp(2 * x);
    if (e2x.isInfinite) return 1.0;
    return (e2x - 1) / (e2x + 1);
  }

  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(_tanh(m.value[i][j]));
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j] * (1 - pow(out.value[i][j], 2));
      }
    }
  }, opName: 'tanhMatrix', cost: numRows * numCols);
  return out;
}