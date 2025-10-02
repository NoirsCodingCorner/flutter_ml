import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Exponential Linear Unit (ELU) activation function.
///
/// Like `LeakyReLU`, ELU is an alternative to `ReLU` that addresses the "Dying ReLU"
/// problem by having a non-zero output for negative inputs.
///
/// Instead of a linear slope, ELU uses a smooth exponential curve for negative
/// values, which can help the optimizer converge more quickly. Its outputs
/// for negative inputs are also negative, which can help push the mean
/// activation closer to zero, improving learning.
///
///
///
/// ### Example
/// ```dart
/// Layer hiddenLayer = DenseLayer(128, activation: ELU(alpha: 1.0));
/// ```
class ELU implements ActivationFunction {
  /// Controls the saturation point for negative inputs. Default is 1.0.
  final double alpha;

  ELU({this.alpha = 1.0});

  /// Applies the ELU function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorELU(input as Tensor<Vector>, alpha: alpha);
  }
}

/// Mathematical operation for the ELU function on a vector.
Tensor<Vector> vectorELU(Tensor<Vector> v, {double alpha = 1.0}) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] > 0 ? v.value[i] : alpha * (exp(v.value[i]) - 1));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  // Derivative is 1 for x > 0. For x <= 0, derivative is alpha * e^x.
  // This can be re-written as: out.value[i] + alpha
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      v.grad[i] += out.grad[i] * (v.value[i] > 0 ? 1.0 : out.value[i] + alpha);
    }
  }, opName: 'elu', cost: N);
  return out;
}