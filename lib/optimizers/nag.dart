import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the Nesterov Accelerated Gradient (NAG) optimizer.
///
/// NAG is an improvement over standard momentum. While standard momentum
/// combines the previous velocity with the current gradient, NAG first makes a
/// "lookahead" step in the direction of the velocity and then calculates the
/// gradient from that future position. This correction prevents the optimizer
/// from overshooting the minimum and often leads to faster convergence.
///
/// ### Analogy 🧠
/// Think of a ball rolling down a hill:
/// - **Momentum** is a heavy ball that builds up speed and rolls past small bumps.
/// - **NAG** is a smarter ball that looks ahead at the slope just before its next
///   move. It can slow down if the ground is about to rise, preventing it from
///   overshooting the bottom of the valley.
///
/// ### Example
/// ```dart
/// var optimizer = NAG(model.parameters, learningRate: 0.01, momentum: 0.9);
/// ```
class NAG extends Optimizer {
  final double momentum;
  late Map<Tensor, dynamic> _v;

  NAG(
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

  /// Performs a single optimization step using an efficient NAG update rule.
  @override
  void step() {
    for (Tensor param in parameters) {
      dynamic v = _v[param]!;
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        Vector vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          double vNew = momentum * vVec[i] + gradVec[i];
          vVec[i] = vNew;
          valVec[i] -= learningRate * (gradVec[i] + momentum * vNew);
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        Matrix vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            double vNew = momentum * vMat[r][c] + gradMat[r][c];
            vMat[r][c] = vNew;
            valMat[r][c] -= learningRate * (gradMat[r][c] + momentum * vNew);
          }
        }
      }
    }
  }
}