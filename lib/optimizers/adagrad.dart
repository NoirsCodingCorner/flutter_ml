import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the Adagrad optimizer.
///
/// Adagrad is an adaptive learning rate optimizer that provides individual
/// learning rates for each parameter. It adapts the rates by dividing by the
/// square root of the sum of all past squared gradients.
///
/// This makes it particularly well-suited for sparse data (like word embeddings),
/// as parameters that are updated infrequently will receive larger updates.
///
/// ### Example
/// ```dart
/// Optimizer optimizer = Adagrad(model.parameters, learningRate: 0.01);
/// ```
class Adagrad extends Optimizer {
  final double epsilon;
  late Map<Tensor, dynamic> _gSquaredSum;

  Adagrad(
      super.parameters, {
        required super.learningRate,
        this.epsilon = 1e-8,
      }) {
    _gSquaredSum = {};
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        _gSquaredSum[param] = List<double>.filled((param.value as Vector).length, 0.0);
      } else if (param.value is Matrix) {
        Matrix sumMatrix = [];
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        for (int i = 0; i < numRows; i++) {
          sumMatrix.add(List<double>.filled(numCols, 0.0));
        }
        _gSquaredSum[param] = sumMatrix;
      }
    }
  }

  /// Performs a single optimization step according to the Adagrad update rule.
  @override
  void step() {
    for (Tensor param in parameters) {
      dynamic gSum = _gSquaredSum[param]!;
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector gSumVec = gSum as Vector;
        for (int i = 0; i < valVec.length; i++) {
          gSumVec[i] += pow(gradVec[i], 2);
          valVec[i] -= learningRate * gradVec[i] / (sqrt(gSumVec[i]) + epsilon);
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix gSumMat = gSum as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            gSumMat[r][c] += pow(gradMat[r][c], 2);
            valMat[r][c] -= learningRate * gradMat[r][c] / (sqrt(gSumMat[r][c]) + epsilon);
          }
        }
      }
    }
  }
}