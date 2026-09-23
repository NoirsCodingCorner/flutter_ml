import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class LayerNormalizationTL extends TapeLayer {
  int numFeatures;
  double epsilon;

  late GPUTensor<Vector> gamma;
  late GPUTensor<Vector> beta;

  LayerNormalizationTL(this.numFeatures, {this.epsilon = 1e-12});

  @override
  String get name {
    return 'LayerNormalizationTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(gamma);
      params.add(beta);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    List<double> ones = <double>[];
    List<double> zeros = <double>[];

    for (int i = 0; i < numFeatures; i = i + 1) {
      ones.add(1.0);
      zeros.add(0.0);
    }

    gamma = GPUTensor<Vector>(ones);
    beta = GPUTensor<Vector>(zeros);

    built = true;
  }

  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      // FIXED: Safetensors uses .weight and .bias instead of .gamma and .beta
      map['$prefix.weight'] = gamma;
      map['$prefix.bias'] = beta;
    }
    return map;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> m = input as GPUTensor<Matrix>;
    int batchSize = m.shape[0];

    List<int> cacheShape = <int>[batchSize];
    GPUTensor<Vector> meanCache = GPUTensor<Vector>.empty(cacheShape);
    GPUTensor<Vector> rstdCache = GPUTensor<Vector>.empty(cacheShape);

    intermediates.add(meanCache);
    intermediates.add(rstdCache);

    GPUTensor<Matrix> out = layerNormMatrixGPU(m, gamma, beta, meanCache, rstdCache, epsilon, tape);
    intermediates.add(out);

    return out;
  }

  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}