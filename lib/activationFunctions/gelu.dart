import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'activation_funciton.dart';

/// An activation function that applies the Gaussian Error Linear Unit (GELU) to a Vector.
///
/// GELU is a high-performing, smooth activation function that is the standard
/// in modern Transformer models like BERT and GPT.
class GELU implements ActivationFunction {
  @override
  Tensor<Vector> call(Tensor<dynamic> input) {
    return vectorGELU(input as Tensor<Vector>);
  }
}

/// Mathematical operation for the GELU function on a vector.
Tensor<Vector> vectorGELU(Tensor<Vector> x) {
  double c1 = 0.044715;
  double c2 = sqrt(2.0 / pi);

  Tensor<Vector> x_pow2 = elementWiseMultiply(x, x);
  Tensor<Vector> x_pow3 = elementWiseMultiply(x_pow2, x);

  Tensor<Vector> inner_term = add(x, scale(x_pow3, c1));

  Tensor<Vector> tanh_out = vectorTanh(scale(inner_term, c2));

  Tensor<Vector> one_plus_tanh = addScalar(tanh_out, 1.0);

  Tensor<Vector> half_x = scale(x, 0.5);

  return elementWiseMultiply(half_x, one_plus_tanh);
}

// Helper operations used by vectorGELU
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


/// An activation function that applies the Gaussian Error Linear Unit (GELU) to a Matrix.
///
/// This version is designed to work on 2D `Matrix` inputs, such as a batch of
/// samples or the output of a self-attention layer.
class GELUMatrix implements ActivationFunction {
  @override
  Tensor<Matrix> call(Tensor<dynamic> input) {
    return matrixGELU(input as Tensor<Matrix>);
  }
}

/// Mathematical operation for the GELU function on a matrix.
Tensor<Matrix> matrixGELU(Tensor<Matrix> m) {
  double c1 = 0.044715;
  double c2 = sqrt(2.0 / pi);

  Tensor<Matrix> m_pow2 = elementWiseMultiplyMatrix(m, m);
  Tensor<Matrix> m_pow3 = elementWiseMultiplyMatrix(m_pow2, m);

  Tensor<Matrix> inner_term = addMatrix(m, scaleMatrix(m_pow3, c1));

  Tensor<Matrix> tanh_out = tanhMatrix(scaleMatrix(inner_term, c2));

  Tensor<Matrix> one_plus_tanh = addScalarToMatrix(tanh_out, Tensor<Scalar>(1.0));

  Tensor<Matrix> half_m = scaleMatrix(m, 0.5);

  return elementWiseMultiplyMatrix(half_m, one_plus_tanh);
}