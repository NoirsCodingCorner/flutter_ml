import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements Stochastic Gradient Descent (SGD) with Momentum.
///
/// This optimizer accelerates standard SGD by adding a "velocity" term, which
/// is an exponentially decaying average of past gradients. This helps the
/// optimizer build speed in consistent directions and dampen oscillations.
///
/// It is a classic and powerful optimizer that is still widely used, especially
/// in computer vision, where it can sometimes lead to better generalization than Adam.
///
/// ### Analogy 🧠
/// Think of a heavy ball rolling down a hill. It builds up momentum, allowing it
/// to roll over small bumps and settle more quickly into the deepest part of a valley.
///
/// ### Example
/// ```dart
/// Optimizer optimizer = Momentum(model.parameters, learningRate: 0.01, momentum: 0.9);
/// ```
class Momentum extends Optimizer {
  final double momentum;
  late Map<Tensor, dynamic> _v;

  Momentum(
      super.parameters, {
        required super.learningRate,
        this.momentum = 0.9,
      }) {
    _v = {};
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        _v[param] = List<double>.filled((param.value as Vector).length, 0.0);
      } else if (param.value is Matrix) {
        Matrix vMatrix = [];
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        for (int i = 0; i < numRows; i++) {
          vMatrix.add(List<double>.filled(numCols, 0.0));
        }
        _v[param] = vMatrix;
      }
    }
  }

  /// Performs a single optimization step using the momentum update rule.
  @override
  void step() {
    for (Tensor param in parameters) {
      dynamic v = _v[param]!;
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          vVec[i] = momentum * vVec[i] + learningRate * gradVec[i];
          valVec[i] -= vVec[i];
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            vMat[r][c] = momentum * vMat[r][c] + learningRate * gradMat[r][c];
            valMat[r][c] -= vMat[r][c];
          }
        }
      }
    }
  }
}