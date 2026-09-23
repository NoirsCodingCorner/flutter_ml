import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'activationFunction.dart';


/// The Mish activation function.
///
/// Mish is a modern, self-gated, and smooth non-monotonic activation function
/// that has achieved state-of-the-art results on a number of computer vision
/// and NLP benchmarks, often outperforming `ReLU` and `Swish`.
///

class MishVector implements ActivationFunction<Vector> {
  @override
  Tensor<Vector> call(Tensor<Vector> input) {
    return mishVector(input);
  }
}

/// Mathematical operation for the Mish function on a vector.
///
/// Built by composing other autograd operations. The backward pass is handled automatically.
class MishMatrix implements ActivationFunction<Matrix> {
  @override
  Tensor<Matrix> call(Tensor<Matrix> input) {
    return mishMatrix(input);
  }
}