import 'tapeLayer.dart';
import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';

class SigmoidMatrixTL extends TapeLayer {
  @override
  String get name => 'SigmoidMatrixTapeLayer';

  @override
  List<GPUTensor> get parameters => <GPUTensor>[];

  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    GPUTensor<Matrix> out = sigmoidMatrixGPU(typedInput, tape);
    intermediates.add(out);
    return out;
  }

  @override
  Map<String, List<dynamic>> getWeights() => <String, List<dynamic>>{};

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}