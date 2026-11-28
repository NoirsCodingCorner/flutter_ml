import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Applies Layer Normalization over a batch of inputs.
/// Standardizes the features for each sample to have a mean of 0 and a variance of 1.
class LayerNormalizationTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'LayerNormalizationTapeLayer';

  int numFeatures;
  double epsilon;

  /// Learnable scaling vector.
  late GPUTensor<Vector> gamma;

  /// Learnable shifting vector.
  late GPUTensor<Vector> beta;

  /// Requires the number of expected features [numFeatures] and a small [epsilon] to prevent division by zero.
  LayerNormalizationTL(this.numFeatures, {this.epsilon = 1e-12});

  /// Returns the trainable [gamma] and [beta] parameters.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(gamma);
      params.add(beta);
    }
    return params;
  }

  /// Allocates VRAM for gamma and beta, initializing to 1.0 and 0.0 respectively.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    List<double> ones = <double>[];
    List<double> zeros = <double>[];

    for (int i = 0; i < numFeatures; i = i + 1) {
      ones.add(1.0);
      zeros.add(0.0);
    }

    gamma = GPUTensor<Vector>(ones, allocTape, sourceOp: '$name:gamma');
    beta = GPUTensor<Vector>(zeros, allocTape, sourceOp: '$name:beta');

    built = true;
  }

  /// Used specifically for SafeTensors compatibility to map weights to standard transformer nomenclature.
  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      map['$prefix.weight'] = gamma;
      map['$prefix.bias'] = beta;
    }
    return map;
  }

  /// Appends [layerNormMatrixGPU] to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    int batchSize = input.shape[0];
    List<int> cacheShape = <int>[batchSize];

    GPUTensor<Vector> meanCache = GPUTensor<Vector>.empty(cacheShape, allocTape,
        requiresGrad: false, sourceOp: '$name:meanCache');
    GPUTensor<Vector> rstdCache = GPUTensor<Vector>.empty(cacheShape, allocTape,
        requiresGrad: false, sourceOp: '$name:rstdCache');

    GPUTensor<Matrix> out = layerNormMatrixGPU(
        input, gamma, beta, meanCache, rstdCache, epsilon, tape, allocTape);

    return out;
  }

  /// Returns a map containing [gamma] and [beta].
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    gamma.dataToCpu();
    beta.dataToCpu();

    wMap['gamma'] = gamma.value;
    wMap['beta'] = beta.value;

    return wMap;
  }

  /// Sets this layer's weights [gamma] and [beta] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];

    List<dynamic> rawGamma = newWeights['gamma']!;
    List<dynamic> rawBeta = newWeights['beta']!;

    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(rawGamma[i] as double);
      betaValues.add(rawBeta[i] as double);
    }

    gamma.pushData(gammaValues);
    beta.pushData(betaValues);
  }
}
