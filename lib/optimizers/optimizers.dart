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