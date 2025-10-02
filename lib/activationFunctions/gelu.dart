import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// The Gaussian Error Linear Unit (GELU) activation function.
///
/// GELU is a high-performing, smooth activation function that is the standard
/// in modern Transformer models like BERT and GPT.
///
/// This implementation uses a common and highly accurate `tanh` approximation.
///
/// ### Example
/// ```dart
/// // A hidden layer in a Transformer-style model.
/// Layer transformerFFN = DenseLayer(2048, activation: GELU());
/// ```
class GELU implements ActivationFunction {
  /// Applies the GELU function element-wise to the input tensor.
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorGELU(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the GELU function on a vector.
///
/// This demonstrates the power of your autograd engine. Instead of calculating
/// the complex derivative by hand, we build the forward pass from existing
/// operations, and the engine handles the backward pass automatically.
Tensor<Vector> vectorGELU(Tensor<Vector> x) {
  double c1 = 0.044715;
  double c2 = sqrt(2.0 / pi);

  // x^3
  Tensor<Vector> x_pow2 = elementWiseMultiply(x, x);
  Tensor<Vector> x_pow3 = elementWiseMultiply(x_pow2, x);

  // (x + 0.044715 * x^3)
  Tensor<Vector> inner_term = add(x, scale(x_pow3, c1));

  // tanh(...)
  Tensor<Vector> tanh_out = vectorTanh(scale(inner_term, c2));

  // 1 + tanh(...)
  Tensor<Vector> one_plus_tanh = addScalar(tanh_out, 1.0);

  // 0.5 * x
  Tensor<Vector> half_x = scale(x, 0.5);

  // 0.5 * x * (1 + tanh(...))
  return elementWiseMultiply(half_x, one_plus_tanh);
}

// You will need to add these two simple scalar helper operations:

Tensor<Vector> scale(Tensor<Vector> v, double s) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) { outValue.add(v.value[i] * s); }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) { v.grad[i] += out.grad[i] * s; }
  }, opName: 'scale', cost: N);
  return out;
}
