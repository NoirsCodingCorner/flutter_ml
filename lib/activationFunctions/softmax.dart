import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// An activation function that applies Softmax to a Vector.
///
/// Softmax converts a vector of real numbers (logits) into a probability
/// distribution where all elements sum to 1. It is the standard activation
/// for the output layer in multi-class classification problems.
///
/// This version is designed to work on a single 1D `Vector` input.
class Softmax implements ActivationFunction {
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return softmax(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the Softmax function on a vector.
Tensor<Vector> softmax(Tensor<Vector> v) {
  double maxVal = -double.infinity;
  for (double val in v.value) {
    if (val > maxVal) {
      maxVal = val;
    }
  }

  Vector exps = [];
  double sumExps = 0.0;
  for (double val in v.value) {
    double expVal = exp(val - maxVal);
    exps.add(expVal);
    sumExps += expVal;
  }

  Vector outValue = [];
  for (double expVal in exps) {
    outValue.add(expVal / sumExps);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < out.value.length; i++) {
      for (int j = 0; j < out.value.length; j++) {
        double delta = (i == j) ? 1.0 : 0.0;
        double jacobian = out.value[i] * (delta - out.value[j]);
        v.grad[j] += out.grad[i] * jacobian;
      }
    }
  }, opName: 'softmax', cost: v.value.length * v.value.length);
  return out;
}

/// An activation function that applies Softmax to each row of a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, where each row
/// is a separate set of logits for a sample in a batch.
class SoftmaxMatrix implements ActivationFunction {
  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return softmaxMatrix(input as Tensor<Matrix>);
  }
}

/// Mathematical operation for applying Softmax to each row of a matrix.
Tensor<Matrix> softmaxMatrix(Tensor<Matrix> m) {
  Matrix outValue = [];
  int numRows = m.value.length;
  int numCols = m.value[0].length;

  for (int i = 0; i < numRows; i++) {
    Vector row = m.value[i];
    double maxVal = -double.infinity;
    for (double val in row) {
      if (val > maxVal) {
        maxVal = val;
      }
    }

    Vector exps = [];
    double sumExps = 0.0;
    for (double val in row) {
      double expVal = exp(val - maxVal);
      exps.add(expVal);
      sumExps += expVal;
    }

    Vector softmaxRow = [];
    for (double expVal in exps) {
      softmaxRow.add(expVal / sumExps);
    }
    outValue.add(softmaxRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node([m], () {
    for (int r = 0; r < numRows; r++) {
      Vector softmaxRow = out.value[r];
      Vector gradRow = out.grad[r];

      double dotProduct = 0;
      for (int i = 0; i < numCols; i++) {
        dotProduct += gradRow[i] * softmaxRow[i];
      }

      for (int j = 0; j < numCols; j++) {
        m.grad[r][j] += softmaxRow[j] * (gradRow[j] - dotProduct);
      }
    }
  }, opName: 'softmax_matrix', cost: numRows * numCols * numCols);
  return out;
}