import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Softmax activation function.
///
/// Softmax converts a vector of real numbers (logits) into a probability
/// distribution. Each element of the output is in the range `(0, 1)`, and
/// all elements sum to 1.
///
/// It is the standard activation function for the **output layer** in
/// **multi-class classification** problems. The output can be interpreted as the
/// model's confidence for each class.
///
/// It is almost always paired with the `CategoricalCrossentropy` loss function.
///
/// ### Example
/// ```dart
/// // An output layer for a 10-class problem (e.g., MNIST digits).
/// Layer outputLayer = DenseLayer(10, activation: Softmax());
/// ```
class Softmax implements ActivationFunction {
  /// Applies the Softmax function to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return softmax(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the Softmax function on a vector.
Tensor<Vector> softmax(Tensor<Vector> v) {
  // Numerically stable softmax: subtract the max value from all elements
  // before exponentiating to prevent overflow.
  double maxVal = v.value.reduce(max);
  Vector exps = [];
  for (double val in v.value) {
    exps.add(exp(val - maxVal));
  }

  double sumExps = exps.reduce((a, b) => a + b);

  Vector outValue = [];
  for (double expVal in exps) {
    outValue.add(expVal / sumExps);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  // The backward pass for Softmax is complex. When combined with Cross-Entropy
  // loss, it simplifies greatly. Here is the standalone backward pass.
  out.creator = Node([v], () {
    for (int i = 0; i < out.value.length; i++) {
      for (int j = 0; j < out.value.length; j++) {
        double delta = (i == j) ? 1.0 : 0.0;
        double jacobian = out.value[i] * (delta - out.value[j]);
        v.grad[j] += out.grad[i] * jacobian;
      }
    }
  }, opName: 'softmax', cost: v.value.length * v.value.length);
  return out;
}