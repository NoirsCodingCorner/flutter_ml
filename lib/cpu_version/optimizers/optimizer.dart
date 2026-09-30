/// Holds and exports all optimization algorithms used for training models.
library;

export 'adagrad.dart';
export 'adam.dart';
export 'adamw.dart';
export 'amsgrad.dart';
export 'nag.dart';
export 'rmsprop.dart';
export 'sgd.dart';
export 'sgdmomentum.dart';

import '../../tensor/tensor.dart';

/// The abstract base class for all optimization algorithms.
/// Tracks a collection of trainable [parameters] and manages optimization steps and gradient resets.
abstract class Optimizer {
  /// The collection of trainable parameter tensors managed by this optimizer.
  List<Tensor<dynamic>> parameters;

  /// The step size factor applied during parameter updates.
  double learningRate;

  /// Base constructor initializing the target [parameters] and [learningRate].
  Optimizer(this.parameters, this.learningRate);

  /// Executes a single optimization step to update all [parameters] in place.
  void step();

  /// Clears gradients by resetting the gradient buffer of every managed tensor in [parameters] to 0.0.
  void zeroGrad() {
    for (int i = 0; i < parameters.length; i = i + 1) {
      parameters[i].zeroGrad();
    }
  }
}
