import '../autogradEngine/tensor.dart';
import 'optimizers.dart';

/// Implements the Stochastic Gradient Descent (SGD) optimizer.
///
/// This is the most fundamental optimization algorithm. At each step, it updates
/// every parameter by moving it in the direction of the negative gradient,
/// scaled by the `learningRate`.
///
/// The update rule is: `parameter = parameter - learningRate * gradient`.
///
/// While simple and effective for many problems, it can be slower to converge
/// than more advanced adaptive optimizers like `Adam` or `RMSprop`.
///
/// ### Example
/// ```dart
/// Optimizer optimizer = SGD(model.parameters, learningRate: 0.01);
/// ```
class SGD extends Optimizer {
  SGD(super.parameters, {required super.learningRate});

  /// Performs a single optimization step using the basic gradient descent rule.
  @override
  void step() {
    for (Tensor param in parameters) {
      if (param.value is Vector) {
        Vector valVec = param.value as Vector;
        Vector gradVec = param.grad as Vector;
        for (int i = 0; i < valVec.length; i++) {
          valVec[i] -= learningRate * gradVec[i];
        }
      } else if (param.value is Matrix) {
        Matrix valMat = param.value as Matrix;
        Matrix gradMat = param.grad as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            valMat[r][c] -= learningRate * gradMat[r][c];
          }
        }
      }
    }
  }
}