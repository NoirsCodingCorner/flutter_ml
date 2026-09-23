

/// Export the activation funcitons of this engine
library;
export 'elu.dart';
export 'leakyRelu.dart';
export 'mish.dart';
export 'relu.dart';
export 'sigmoid.dart';
export 'silu.dart';
export 'softmax.dart';


import '../../tensor/tensor.dart';

/// The abstract base class (or interface) for all activation functions.
///
/// The purpose of this class is to define a common contract that all
/// activation functions, such as `ReLU`, `Sigmoid`, or `Tanh`, must follow.
///
/// By having this common interface, layers like `DenseLayer` can be written to
/// work with any activation function, making the framework modular and easy to
/// extend with new, custom activations.
///
/// ### Example
/// ```dart
/// // Both layers accept an object that implements ActivationFunction.
/// Layer denseWithReLU = DenseLayer(64, activation: ReLuVector());
/// Layer denseWithTanh = DenseLayerMatrix(64, activation: TanhMatrix());
/// ```
abstract class ActivationFunction<T> {
  Tensor<T> call(Tensor<T> input);
}