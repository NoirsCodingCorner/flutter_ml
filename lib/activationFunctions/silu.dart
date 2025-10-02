import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Sigmoid-weighted Linear Unit (SiLU) activation function, also known as Swish.
///
/// This is a smooth, non-monotonic function that often outperforms ReLU on deeper
/// models. It is "self-gated," as it uses the sigmoid function to gate the input.
///
/// The function is defined as `$f(x) = x \cdot \sigma(x)$`, where `$\sigma$` is the sigmoid function.
///
///
///
/// ### Example
/// ```dart
/// Layer hiddenLayer = DenseLayer(128, activation: Swish());
/// ```
class Swish implements ActivationFunction {
  /// Applies the Swish function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorSwish(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the Swish function on a vector.
Tensor<Vector> vectorSwish(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  Vector sigmoids = []; // Store sigmoid values for the backward pass

  for (int i = 0; i < N; i++) {
    double sigVal = 1.0 / (1.0 + exp(-v.value[i]));
    sigmoids.add(sigVal);
    outValue.add(v.value[i] * sigVal);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  // The derivative of swish(x) is: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      double sigVal = sigmoids[i];
      double derivative = sigVal * (1 + v.value[i] * (1 - sigVal));
      v.grad[i] += out.grad[i] * derivative;
    }
  }, opName: 'swish', cost: N * 2); // Roughly 2 ops per element
  return out;
}