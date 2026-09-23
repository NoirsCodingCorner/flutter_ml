
/// Export the optimizer functions
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
abstract class Optimizer {
  List<Tensor<dynamic>> parameters;
  double learningRate;

  Optimizer(this.parameters, {required this.learningRate});

  void step();

  void zeroGrad() {
    for (int i = 0; i < parameters.length; i = i + 1) {
      parameters[i].zeroGrad();
    }
  }
}