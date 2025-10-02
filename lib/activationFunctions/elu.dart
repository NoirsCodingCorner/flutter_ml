import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// An activation function that applies the Exponential Linear Unit (ELU) to a Vector.
///
/// ELU is an alternative to `ReLU` that has a small negative value for negative
/// inputs, which can help prevent the "Dying ReLU" problem and speed up learning.
///
/// The function is defined as `$f(x) = x` if `$x > 0$`, and `$f(x) = \alpha(e^x - 1)$` if `$x \le 0$`.
class ELU implements ActivationFunction {
  /// Controls the saturation point for negative inputs.
  final double alpha;

  ELU({this.alpha = 1.0});

  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorELU(input as Tensor<Vector>, alpha: alpha);
  }
}

/// Mathematical operation for the ELU function on a vector.
Tensor<Vector> vectorELU(Tensor<Vector> v, {double alpha = 1.0}) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] > 0 ? v.value[i] : alpha * (exp(v.value[i]) - 1));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      v.grad[i] += out.grad[i] * (v.value[i] > 0 ? 1.0 : out.value[i] + alpha);
    }
  }, opName: 'elu', cost: N);
  return out;
}

/// An activation function that applies the Exponential Linear Unit (ELU) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, applying the ELU
/// function to each element independently.
class ELUMatrix implements ActivationFunction {
  /// Controls the saturation point for negative inputs.
  final double alpha;

  ELUMatrix({this.alpha = 1.0});

  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return matrixELU(input as Tensor<Matrix>, alpha: alpha);
  }
}

/// Mathematical operation for the ELU function on a matrix.
Tensor<Matrix> matrixELU(Tensor<Matrix> m, {double alpha = 1.0}) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] > 0 ? m.value[i][j] : alpha * (exp(m.value[i][j]) - 1));
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j] * (m.value[i][j] > 0 ? 1.0 : out.value[i][j] + alpha);
      }
    }
  }, opName: 'elu_matrix', cost: numRows * numCols);
  return out;
}