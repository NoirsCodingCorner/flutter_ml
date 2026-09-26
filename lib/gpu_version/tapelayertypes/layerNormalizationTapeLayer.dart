import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

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

  /// Persistent Cache for Static Unrolling
  int cacheBatchSize = -1;
  GPUTensor<Vector>? cachedMean;
  GPUTensor<Vector>? cachedRstd;
  GPUTensor<Matrix>? cachedOut;

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
  void build(GPUTensor<Matrix> input) {
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
  /// Persistently caches intermediate tensors to avoid VRAM reallocation between batches.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int batchSize = input.shape[0];

    if (cacheBatchSize != batchSize) {
      if (cachedMean != null) cachedMean!.free();
      if (cachedRstd != null) cachedRstd!.free();
      if (cachedOut != null) cachedOut!.free();

      List<int> cacheShape = <int>[batchSize];
      cachedMean = GPUTensor<Vector>.empty(cacheShape);
      cachedRstd = GPUTensor<Vector>.empty(cacheShape);
      cachedOut = null;
      cacheBatchSize = batchSize;
    }

    // Reuse persistent buffers, bypassing the external intermediates list
    cachedOut = layerNormMatrixGPU(
        input,
        gamma,
        beta,
        cachedMean!,
        cachedRstd!,
        epsilon,
        tape,
        outTensor: cachedOut
    );

    return cachedOut!;
  }

  /// Clears the gradients of all statically cached intermediate tensors.
  /// The optimizer handles gamma and beta.
  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
    // Note: Technically mean and rstd don't accumulate standard backprop gradients from the loss,
    // but zeroing them here is safe and thorough if the kernel ever uses their buffers.
    if (cachedMean != null) {
      cachedMean!.zeroGrad(tape);
    }
    if (cachedRstd != null) {
      cachedRstd!.zeroGrad(tape);
    }
  }

  /// Frees VRAM for parameters and persistently cached intermediate tensors.
  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
    }
    if (cachedMean != null) cachedMean!.free();
    if (cachedRstd != null) cachedRstd!.free();
    if (cachedOut != null) cachedOut!.free();
  }

  /// Returns a map containing [gamma] and [beta].
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    gamma.toCpu();
    beta.toCpu();

    wMap['gamma'] = gamma.value;
    wMap['beta'] = beta.value;

    return wMap;
  }

  /// Sets this layer's weights [gamma] and [beta] from a map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      gamma.free();
      beta.free();
    }

    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];

    List<dynamic> rawGamma = newWeights['gamma']!;
    List<dynamic> rawBeta = newWeights['beta']!;

    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(rawGamma[i] as double);
      betaValues.add(rawBeta[i] as double);
    }

    gamma = GPUTensor<Vector>(gammaValues);
    beta = GPUTensor<Vector>(betaValues);

    built = true;
  }
}