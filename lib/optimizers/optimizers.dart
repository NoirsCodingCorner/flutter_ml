import 'dart:math';

import '../autogradEngine/tensor.dart';

/// The abstract base class for all optimization algorithms.
///
/// An `Optimizer`'s role is to hold a model's trainable `parameters` and update
/// them according to a specific algorithm (e.g., `SGD`, `Adam`) using the
/// gradients computed during the backward pass.
///
/// This abstraction allows the training loop to remain generic; you can easily
/// swap out one optimizer for another without changing the training code.
///
/// ### Training Workflow
/// The typical training loop sequence is:
/// 1. `loss.backward()` - Compute gradients for all parameters.
/// 2. `optimizer.step()` - Update all parameters using the gradients.
/// 3. `optimizer.zeroGrad()` - Reset all gradients to zero for the next iteration.
///
/// ### Example
/// ```dart
/// // 1. Collect model parameters and create an optimizer.
/// Optimizer optimizer = Adam(myModel.parameters, learningRate: 0.001);
///
/// // 2. Inside the training loop...
/// for (var sample in dataset) {
///   var loss = myModel.forward(sample.input);
///   loss.backward();
///   optimizer.step();
///   optimizer.zeroGrad();
/// }
/// ```
abstract class Optimizer {
  /// The list of model parameters (weights and biases) that this optimizer will update.
  final List<Tensor> parameters;

  /// The step size for the gradient updates.
  final double learningRate;

  Optimizer(this.parameters, {required this.learningRate});

  /// Performs a single optimization step (parameter update).
  ///
  /// Subclasses **must** implement this method to define their specific update rule
  /// (e.g., the standard gradient descent update, or the more complex Adam update).
  void step();

  /// Resets the gradients of all parameters to zero.
  ///
  /// This must be called at the end of each training iteration to prevent the
  /// gradients from one batch from accumulating into the next.
  void zeroGrad() {
    for (var param in parameters) {
      param.zeroGrad();
    }
  }
}

class SGD extends Optimizer {
  SGD(super.parameters, {required super.learningRate});

  @override
  void step() {
    for (var param in parameters) {
      if (param.value is Vector) {
        for (int i = 0; i < (param.value as Vector).length; i++) {
          (param.value as Vector)[i] -= learningRate * (param.grad as Vector)[i];
        }
      } else if (param.value is Matrix) {
        for (int r = 0; r < (param.value as Matrix).length; r++) {
          for (int c = 0; c < (param.value as Matrix)[0].length; c++) {
            (param.value as Matrix)[r][c] -= learningRate * (param.grad as Matrix)[r][c];
          }
        }
      }
    }
  }

  @override
  void zeroGrad() {
    for (var param in parameters) {
      param.zeroGrad(); // Use the zeroGrad method you already wrote!
    }
  }
}


class RMSprop extends Optimizer {
  /// The decay rate for the moving average of squared gradients.
  /// A value close to 1 results in a slower-changing average.
  final double beta;

  /// A small constant for numerical stability to prevent division by zero.
  final double epsilon;

  // A map to store the moving average of squared gradients for each parameter.
  late Map<Tensor, dynamic> _s;
  /// Implements the RMSprop optimizer.
  ///
  /// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimizer.
  /// It maintains a moving average of the square of gradients for each parameter,
  /// effectively decreasing the learning rate for parameters with large, volatile
  /// gradients and increasing it for those with small, consistent gradients.
  ///
  /// It's often a good choice for recurrent neural networks.
  ///
  /// ### Example
  /// ```dart
  /// var optimizer = RMSprop(model.parameters, learningRate: 0.001);
  /// optimizer.step();
  /// ```
  RMSprop(
      super.parameters, {
        required super.learningRate,
        this.beta = 0.99,
        this.epsilon = 1e-8,
      }) {
    // Initialize the cache for each parameter with zeros.
    _s = {};
    for (var param in parameters) {
      if (param.value is Vector) {
        _s[param] = List<double>.filled((param.value as Vector).length, 0.0);
      } else if (param.value is Matrix) {
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        _s[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
      }
    }
  }

  /// Performs a single optimization step.
  ///
  /// This method updates each parameter according to the RMSprop update rule:
  /// 1. Update the moving average of squared gradients.
  /// 2. Update the parameter by dividing the learning rate by the root of this average.
  @override
  void step() {
    for (var param in parameters) {
      var s = _s[param]!;
      if (param.value is Vector) {
        var valVec = param.value as Vector;
        var gradVec = param.grad as Vector;
        var sVec = s as Vector;
        for (int i = 0; i < valVec.length; i++) {
          // Update the moving average of squared gradients.
          sVec[i] = beta * sVec[i] + (1 - beta) * pow(gradVec[i], 2);
          // Update the parameter.
          valVec[i] -= learningRate * gradVec[i] / (sqrt(sVec[i]) + epsilon);
        }
      } else if (param.value is Matrix) {
        var valMat = param.value as Matrix;
        var gradMat = param.grad as Matrix;
        var sMat = s as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            // Update the moving average of squared gradients.
            sMat[r][c] = beta * sMat[r][c] + (1 - beta) * pow(gradMat[r][c], 2);
            // Update the parameter.
            valMat[r][c] -= learningRate * gradMat[r][c] / (sqrt(sMat[r][c]) + epsilon);
          }
        }
      }
    }
  }
}
class NAG extends Optimizer {
  /// The momentum factor (gamma). It determines how much of the previous
  /// update is carried over to the current step. Typically set to 0.9.
  final double momentum;

  // A map to store the velocity vector for each parameter.
  late Map<Tensor, dynamic> _v;

  /// Implements the Nesterov Accelerated Gradient (NAG) optimizer.
  ///
  /// NAG is an improvement over standard momentum. While standard momentum
  /// combines the previous velocity with the current gradient, NAG first makes a
  /// "lookahead" step in the direction of the velocity and then calculates the
  /// gradient from that future position. This correction prevents the optimizer
  /// from overshooting the minimum and often leads to faster convergence.
  ///
  /// ### Analogy
  /// Think of a ball rolling down a hill:
  /// - **Momentum** is a heavy ball that builds up speed and rolls past small bumps.
  /// - **NAG** is a smarter ball that looks ahead at the slope just before its next
  ///   move. It can slow down if the ground is about to rise, preventing it from
  ///   overshooting the bottom of the valley.
  ///
  /// ### Example
  /// ```dart
  /// var optimizer = NAG(model.parameters, learningRate: 0.01, momentum: 0.9);
  /// optimizer.step();
  /// ```

  NAG(
      super.parameters, {
        required super.learningRate,
        this.momentum = 0.9,
      }) {
    // Initialize the velocity buffer for each parameter with zeros.
    _v = {};
    for (var param in parameters) {
      if (param.value is Vector) {
        _v[param] = List<double>.filled((param.value as Vector).length, 0.0);
      } else if (param.value is Matrix) {
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        _v[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
      }
    }
  }

  /// Performs a single optimization step using the NAG update rule.
  @override
  void step() {
    for (var param in parameters) {
      var v = _v[param]!;
      if (param.value is Vector) {
        var valVec = param.value as Vector;
        var gradVec = param.grad as Vector;
        var vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          // 1. Calculate the velocity update (same as standard momentum).
          var vNew = momentum * vVec[i] + gradVec[i];
          // 2. Store the new velocity.
          vVec[i] = vNew;
          // 3. Apply the Nesterov correction: use the current gradient AND the new velocity.
          valVec[i] -= learningRate * (gradVec[i] + momentum * vNew);
        }
      } else if (param.value is Matrix) {
        var valMat = param.value as Matrix;
        var gradMat = param.grad as Matrix;
        var vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            // 1. Calculate the velocity update.
            var vNew = momentum * vMat[r][c] + gradMat[r][c];
            // 2. Store the new velocity.
            vMat[r][c] = vNew;
            // 3. Apply the Nesterov correction.
            valMat[r][c] -= learningRate * (gradMat[r][c] + momentum * vNew);
          }
        }
      }
    }
  }
}

class Adam extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;

  // Timestep counter
  int _t = 0;

  // Maps to store the 1st and 2nd moment vectors for each parameter
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
    // Initialize moments for each parameter with zeros
    for (var param in parameters) {
      if (param.value is Vector) {
        int size = (param.value as Vector).length;
        _m[param] = List<double>.filled(size, 0.0);
        _v[param] = List<double>.filled(size, 0.0);
      } else if (param.value is Matrix) {
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        _m[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
        _v[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
      }
    }
  }

  @override
  void step() {
    _t++; // Increment timestep
    for (var param in parameters) {
      var m = _m[param]!;
      var v = _v[param]!;

      if (param.value is Vector) {
        var valVec = param.value as Vector;
        var gradVec = param.grad as Vector;
        var mVec = m as Vector;
        var vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          // Update biased 1st and 2nd moment estimates
          mVec[i] = beta1 * mVec[i] + (1 - beta1) * gradVec[i];
          vVec[i] = beta2 * vVec[i] + (1 - beta2) * pow(gradVec[i], 2);

          // Compute bias-corrected 1st and 2nd moment estimates
          double mHat = mVec[i] / (1 - pow(beta1, _t));
          double vHat = vVec[i] / (1 - pow(beta2, _t));

          // Update parameters
          valVec[i] -= learningRate * mHat / (sqrt(vHat) + epsilon);
        }
      } else if (param.value is Matrix) {
        var valMat = param.value as Matrix;
        var gradMat = param.grad as Matrix;
        var mMat = m as Matrix;
        var vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            // Update biased 1st and 2nd moment estimates
            mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
            vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);

            // Compute bias-corrected 1st and 2nd moment estimates
            double mHat = mMat[r][c] / (1 - pow(beta1, _t));
            double vHat = vMat[r][c] / (1 - pow(beta2, _t));

            // Update parameters
            valMat[r][c] -= learningRate * mHat / (sqrt(vHat) + epsilon);
          }
        }
      }
    }
  }
}
class AdamW extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;

  /// The weight decay factor (lambda). It penalizes large weights to
  /// prevent overfitting. A typical value is 0.01.
  final double weightDecay;

  // Timestep counter.
  int _t = 0;

  // Maps to store the 1st and 2nd moment vectors for each parameter.
  late Map<Tensor, dynamic> _m;
  late Map<Tensor, dynamic> _v;
  /// Implements the AdamW optimizer.
  ///
  /// AdamW is a variant of the Adam optimizer that improves its handling of
  /// L2 regularization, also known as weight decay. In standard Adam, weight decay
  /// is often implemented incorrectly, causing it to be less effective. AdamW
  /// decouples the weight decay from the gradient-based update, applying it
  /// directly to the weights.
  ///
  /// This often leads to better model generalization and performance. It has become
  /// the default optimizer for training large models like Transformers (e.g., GPT, BERT).
  ///
  /// ### Analogy 🧠
  /// Think of training as driving a car and weight decay as a tax on your position:
  /// - **Adam with L2 regularization:** The tax is bundled with your fuel cost. When you
  ///   accelerate hard (large gradients), your tax also increases, which isn't logical.
  /// - **AdamW:** The tax is paid separately based on your position, regardless of
  ///   your speed. This is a more stable and effective way to regularize.
  ///
  /// ### Example
  /// ```dart
  /// var optimizer = AdamW(model.parameters, learningRate: 0.001, weightDecay: 0.01);
  /// optimizer.step();
  /// ```

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
    // Initialize moments for each parameter with zeros.
    for (var param in parameters) {
      if (param.value is Vector) {
        int size = (param.value as Vector).length;
        _m[param] = List<double>.filled(size, 0.0);
        _v[param] = List<double>.filled(size, 0.0);
      } else if (param.value is Matrix) {
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        _m[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
        _v[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
      }
    }
  }

  @override
  void step() {
    _t++; // Increment timestep.
    for (var param in parameters) {
      var m = _m[param]!;
      var v = _v[param]!;

      if (param.value is Vector) {
        var valVec = param.value as Vector;
        var gradVec = param.grad as Vector;
        var mVec = m as Vector;
        var vVec = v as Vector;
        for (int i = 0; i < valVec.length; i++) {
          // Update biased 1st and 2nd moment estimates.
          mVec[i] = beta1 * mVec[i] + (1 - beta1) * gradVec[i];
          vVec[i] = beta2 * vVec[i] + (1 - beta2) * pow(gradVec[i], 2);

          // Compute bias-corrected 1st and 2nd moment estimates.
          double mHat = mVec[i] / (1 - pow(beta1, _t));
          double vHat = vVec[i] / (1 - pow(beta2, _t));

          // **THE ADAMW MODIFICATION IS HERE**
          // 1. Perform the standard Adam update.
          valVec[i] -= learningRate * mHat / (sqrt(vHat) + epsilon);
          // 2. Apply the decoupled weight decay directly to the parameter.
          valVec[i] -= learningRate * weightDecay * valVec[i];
        }
      } else if (param.value is Matrix) {
        var valMat = param.value as Matrix;
        var gradMat = param.grad as Matrix;
        var mMat = m as Matrix;
        var vMat = v as Matrix;
        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            // Update biased 1st and 2nd moment estimates.
            mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
            vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);

            // Compute bias-corrected 1st and 2nd moment estimates.
            double mHat = mMat[r][c] / (1 - pow(beta1, _t));
            double vHat = vMat[r][c] / (1 - pow(beta2, _t));

            // **THE ADAMW MODIFICATION IS HERE**
            // 1. Perform the standard Adam update.
            valMat[r][c] -= learningRate * mHat / (sqrt(vHat) + epsilon);
            // 2. Apply the decoupled weight decay directly to the parameter.
            valMat[r][c] -= learningRate * weightDecay * valMat[r][c];
          }
        }
      }
    }
  }
}

/// Implements the AMSGrad optimizer.
///
/// AMSGrad is a variant of the Adam optimizer that uses the maximum of past
/// squared gradients rather than the exponential average to update the parameters.
/// This modification addresses a potential convergence issue in Adam, making AMSGrad
/// more stable and robust, especially in complex optimization landscapes.
///
/// ### Example
/// ```dart
/// var optimizer = AMSGrad(model.parameters, learningRate: 0.001);
/// optimizer.step();
/// ```
class AMSGrad extends Optimizer {
  final double beta1;
  final double beta2;
  final double epsilon;

  // Timestep counter
  int _t = 0;

  // Maps to store the 1st moment, 2nd moment, and max 2nd moment vectors
  late Map<Tensor, dynamic> _m;
  late Map<Tensor, dynamic> _v;
  late Map<Tensor, dynamic> _vHat; // NEW: Stores the max of v

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
    // Initialize moments for each parameter with zeros
    for (var param in parameters) {
      if (param.value is Vector) {
        int size = (param.value as Vector).length;
        _m[param] = List<double>.filled(size, 0.0);
        _v[param] = List<double>.filled(size, 0.0);
        _vHat[param] = List<double>.filled(size, 0.0);
      } else if (param.value is Matrix) {
        int numRows = (param.value as Matrix).length;
        int numCols = numRows > 0 ? (param.value as Matrix)[0].length : 0;
        _m[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
        _v[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
        _vHat[param] = List.generate(numRows, (_) => List<double>.filled(numCols, 0.0));
      }
    }
  }

  @override
  void step() {
    _t++; // Increment timestep
    for (var param in parameters) {
      var m = _m[param]!;
      var v = _v[param]!;
      var vHatBuffer = _vHat[param]!;

      if (param.value is Vector) {
        var valVec = param.value as Vector;
        var gradVec = param.grad as Vector;
        var mVec = m as Vector;
        var vVec = v as Vector;
        var vHatVec = vHatBuffer as Vector;

        for (int i = 0; i < valVec.length; i++) {
          // Update biased 1st and 2nd moment estimates (same as Adam)
          mVec[i] = beta1 * mVec[i] + (1 - beta1) * gradVec[i];
          vVec[i] = beta2 * vVec[i] + (1 - beta2) * pow(gradVec[i], 2);

          // **THE AMSGRAD MODIFICATION IS HERE**
          // Update the maximum of the 2nd moment estimates
          vHatVec[i] = max(vHatVec[i], vVec[i]);

          // Compute bias-corrected 1st moment estimate
          double mHat = mVec[i] / (1 - pow(beta1, _t));

          // Update parameters using the max of 2nd moments (vHat)
          valVec[i] -= learningRate * mHat / (sqrt(vHatVec[i]) + epsilon);
        }
      } else if (param.value is Matrix) {
        var valMat = param.value as Matrix;
        var gradMat = param.grad as Matrix;
        var mMat = m as Matrix;
        var vMat = v as Matrix;
        var vHatMat = vHatBuffer as Matrix;

        for (int r = 0; r < valMat.length; r++) {
          for (int c = 0; c < valMat[0].length; c++) {
            // Update biased 1st and 2nd moment estimates (same as Adam)
            mMat[r][c] = beta1 * mMat[r][c] + (1 - beta1) * gradMat[r][c];
            vMat[r][c] = beta2 * vMat[r][c] + (1 - beta2) * pow(gradMat[r][c], 2);

            // **THE AMSGRAD MODIFICATION IS HERE**
            // Update the maximum of the 2nd moment estimates
            vHatMat[r][c] = max(vHatMat[r][c], vMat[r][c]);

            // Compute bias-corrected 1st moment estimate
            double mHat = mMat[r][c] / (1 - pow(beta1, _t));

            // Update parameters using the max of 2nd moments (vHat)
            valMat[r][c] -= learningRate * mHat / (sqrt(vHatMat[r][c]) + epsilon);
          }
        }
      }
    }
  }
}