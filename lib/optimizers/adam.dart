import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

class Adam extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;
  int _t = 0;
  late Map<Tensor, dynamic> _m;
  late Map<Tensor, dynamic> _v;

  Adam(
      super.parameters, {
        required super.learningRate,
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-8,
      }) {
    _m = {};
    _v = {};
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        int size = (param.value as Vector).length;
        _m[param] = List<double>.filled(size, 0.0);
        _v[param] = List<double>.filled(size, 0.0);
      } else if (param.value is Matrix) {
        Matrix mMatrix = [];
        Matrix vMatrix = [];
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        for (int i = 0; i < numRows; i++) {
          mMatrix.add(List<double>.filled(numCols, 0.0));
          vMatrix.add(List<double>.filled(numCols, 0.0));
        }
        _m[param] = mMatrix;
        _v[param] = vMatrix;
      }
    }
  }

  @override
  void step() {
    _t++;
    for (Tensor param in parameters) {
      dynamic m = _m[param]!;
      dynamic v = _v[param]!;
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector mVec = m as Vector;
        Vector vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          mVec[i] = beta1 * mVec[i] + (1 - beta1) * gradVec[i];
          vVec[i] = beta2 * vVec[i] + (1 - beta2) * pow(gradVec[i], 2);
          double mHat = mVec[i] / (1 - pow(beta1, _t));
          double vHat = vVec[i] / (1 - pow(beta2, _t));
          valVec[i] -= learningRate * mHat / (sqrt(vHat) + epsilon);
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix mMat = m as Matrix;
        Matrix vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
            vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);
            double mHat = mMat[r][c] / (1 - pow(beta1, _t));
            double vHat = vMat[r][c] / (1 - pow(beta2, _t));
            valMat[r][c] -= learningRate * mHat / (sqrt(vHat) + epsilon);
          }
        }
      }
    }
  }
}

/// Implements the Adam optimizer for 2D Matrix parameters.
///
/// This version is specifically for updating `Tensor<Matrix>` parameters, such
/// as the weight matrices in `DenseLayer` or kernels in `Conv2DLayer`.
class AdamMatrix extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;

  int _t = 0;
  late Map<Tensor, Matrix> _m;
  late Map<Tensor, Matrix> _v;

  AdamMatrix(
      super.parameters, {
        required super.learningRate,
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-8,
      }) {
    _m = {};
    _v = {};
    for (Tensor param in parameters) {
      Matrix mMatrix = [];
      Matrix vMatrix = [];
      int numRows = (param.value as Matrix).length;
      int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
      for (int i = 0; i < numRows; i++) {
        mMatrix.add(List<double>.filled(numCols, 0.0));
        vMatrix.add(List<double>.filled(numCols, 0.0));
      }
      _m[param] = mMatrix;
      _v[param] = vMatrix;
    }
  }

  @override
  void step() {
    _t++;
    for (Tensor param in parameters) {
      Matrix valMat = param.value as Matrix;
      Matrix gradMat = param.grad as Matrix;
      Matrix mMat = _m[param]!;
      Matrix vMat = _v[param]!;

      for (int r = 0; r < valMat.length; r++) {
        for (int c = 0; c < valMat[0].length; c++) {
          mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
          vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);

          double mHat = mMat[r][c] / (1 - pow(beta1, _t));
          double vHat = vMat[r][c] / (1 - pow(beta2, _t));

          valMat[r][c] -= learningRate * mHat / (sqrt(vHat) + epsilon);
        }
      }
    }
  }
}