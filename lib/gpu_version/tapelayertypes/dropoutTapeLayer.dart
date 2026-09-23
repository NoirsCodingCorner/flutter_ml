import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class DropoutTL extends TapeLayer {
  double rate;
  bool isTraining;

  DropoutTL(this.rate, {this.isTraining = true});

  @override
  String get name {
    return 'DropoutTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    // If we are inferencing or the rate is 0, just pass the tensor right through
    if (isTraining == false || rate == 0.0) {
      return input;
    }

    // Call the native CUDA operation
    return dropoutGPU(input, rate, tape);
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> weights) {}
}

class DropoutMatrixTL extends TapeLayer {
  double rate;
  bool isTraining;

  DropoutMatrixTL(this.rate, {this.isTraining = true});

  @override
  String get name {
    return 'DropoutMatrixTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  @override
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    // STRICT TYPING: Cast input so dropoutGPU infers T = Matrix.
    // This entirely prevents the GPUTensor<dynamic> contagion.
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    if (isTraining == false || rate == 0.0) {
      return typedInput;
    }

    return dropoutGPU<Matrix>(typedInput, rate, tape);
  }

  @override
  void free() {}

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}