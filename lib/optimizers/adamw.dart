import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the AdamW optimizer.
///
/// AdamW is a variant of the Adam optimizer that improves its handling of
/// L2 regularization (weight decay). In standard Adam, weight decay is
/// coupled with the adaptive learning rate, which can lead to suboptimal
/// performance. AdamW **decouples** the weight decay from the gradient-based
/// update, applying it directly to the weights.
///
/// This often leads to better model generalization and has become the default
/// optimizer for training large models like Transformers.
///
/// ### Analogy 🧠
/// Think of training as driving a car and weight decay as a tax:
/// - **Adam:** The tax is bundled with your fuel cost. When you accelerate hard
///   (large gradients), your tax also increases.
/// - **AdamW:** The tax is paid separately, based only on your weight values,
///   regardless of your speed (gradient).
///
/// ### Example
/// ```dart
/// var optimizer = AdamW(model.parameters, learningRate: 0.001, weightDecay: 0.01);
/// ```
class AdamW extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;
  final double weightDecay;

  int _t = 0;
  late Map<Tensor, dynamic> _m;
  late Map<Tensor, dynamic> _v;

  AdamW(
      super.parameters, {
        required super.learningRate,
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-8,
        this.weightDecay = 0.01,
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

  /// Performs a single optimization step according to the AdamW update rule.
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
          valVec[i] -= learningRate * weightDecay * valVec[i];
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
            valMat[r][c] -= learningRate * weightDecay * valMat[r][c];
          }
        }
      }
    }
  }
}