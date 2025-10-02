import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the RMSprop optimizer.
///
/// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimizer.
/// It maintains a moving average of the square of gradients for each parameter.
/// This adapts the learning rate on a per-parameter basis, decreasing it for
/// parameters with large gradients and increasing it for those with small gradients.
///
/// It's often a good choice for recurrent neural networks or when dealing with
/// noisy gradients.
///
/// ### Example
/// ```dart
/// var optimizer = RMSprop(model.parameters, learningRate: 0.001);
/// ```
class RMSprop extends Optimizer {
  final double beta;
  final double epsilon;
  late Map<Tensor, dynamic> _s;

  RMSprop(
      super.parameters, {
        required super.learningRate,
        this.beta = 0.99,
        this.epsilon = 1e-8,
      }) {
    _s = {};
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        _s[param] = List<double>.filled((param.value as Vector).length, 0.0);
      } else if (param.value is Matrix) {
        Matrix sMatrix = [];
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        for (int i = 0; i < numRows; i++) {
          sMatrix.add(List<double>.filled(numCols, 0.0));
        }
        _s[param] = sMatrix;
      }
    }
  }

  /// Performs a single optimization step according to the RMSprop update rule.
  @override
  void step() {
    for (Tensor param in parameters) {
      dynamic s = _s[param]!;
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector sVec = s as Vector;
        for (int i = 0; i < valVec.length; i++) {
          sVec[i] = beta * sVec[i] + (1 - beta) * pow(gradVec[i], 2);
          valVec[i] -= learningRate * gradVec[i] / (sqrt(sVec[i]) + epsilon);
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix sMat = s as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            sMat[r][c] = beta * sMat[r][c] + (1 - beta) * pow(gradMat[r][c], 2);
            valMat[r][c] -= learningRate * gradMat[r][c] / (sqrt(sMat[r][c]) + epsilon);
          }
        }
      }
    }
  }
}