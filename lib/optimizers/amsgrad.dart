import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the AMSGrad optimizer.
///
/// AMSGrad is a variant of the Adam optimizer that aims to fix a potential
/// convergence issue. It achieves this by using the **maximum** of past squared
/// gradients (`v_hat`) for normalization, rather than just the exponential average (`v`).
///
/// This ensures that the adaptive learning rate is non-increasing, which can
/// improve stability and guarantee convergence in some scenarios where Adam might fail.
///
/// ### Example
/// ```dart
/// var optimizer = AMSGrad(model.parameters, learningRate: 0.001);
/// ```
class AMSGrad extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;

  int _t = 0;
  late Map<Tensor, dynamic> _m;
  late Map<Tensor, dynamic> _v;
  late Map<Tensor, dynamic> _vHat;

  AMSGrad(
      super.parameters, {
        required super.learningRate,
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-8,
      }) {
    _m = {};
    _v = {};
    _vHat = {};
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        int size = (param.value as Vector).length;
        _m[param] = List<double>.filled(size, 0.0);
        _v[param] = List<double>.filled(size, 0.0);
        _vHat[param] = List<double>.filled(size, 0.0);
      } else if (param.value is Matrix) {
        Matrix mMatrix = [];
        Matrix vMatrix = [];
        Matrix vHatMatrix = [];
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        for (int i = 0; i < numRows; i++) {
          mMatrix.add(List<double>.filled(numCols, 0.0));
          vMatrix.add(List<double>.filled(numCols, 0.0));
          vHatMatrix.add(List<double>.filled(numCols, 0.0));
        }
        _m[param] = mMatrix;
        _v[param] = vMatrix;
        _vHat[param] = vHatMatrix;
      }
    }
  }

  /// Performs a single optimization step according to the AMSGrad update rule.
  @override
  void step() {
    _t++;
    for (Tensor param in parameters) {
      dynamic m = _m[param]!;
      dynamic v = _v[param]!;
      dynamic vHatBuffer = _vHat[param]!;

      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector mVec = m as Vector;
        Vector vVec = v as Vector;
        Vector vHatVec = vHatBuffer as Vector;

        for (int i = 0; i < valVec.length; i++) {
          mVec[i] = beta1 * mVec[i] + (1 - beta1) * gradVec[i];
          vVec[i] = beta2 * vVec[i] + (1 - beta2) * pow(gradVec[i], 2);
          vHatVec[i] = max(vHatVec[i], vVec[i]);

          double mHat = mVec[i] / (1 - pow(beta1, _t));

          valVec[i] -= learningRate * mHat / (sqrt(vHatVec[i]) + epsilon);
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix mMat = m as Matrix;
        Matrix vMat = v as Matrix;
        Matrix vHatMat = vHatBuffer as Matrix;

        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
            vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);
            vHatMat[r][c] = max(vHatMat[r][c], vMat[r][c]);

            double mHat = mMat[r][c] / (1 - pow(beta1, _t));

            valMat[r][c] -= learningRate * mHat / (sqrt(vHatMat[r][c]) + epsilon);
          }
        }
      }
    }
  }
}