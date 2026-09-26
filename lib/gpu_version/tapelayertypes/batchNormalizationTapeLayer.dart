import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Applies Batch Normalization over a 1D input vector to enforce a mean of 0 and a variance of 1.
class BatchNorm1DTL extends TapeLayer<Vector, Vector> {
  String get name => 'BatchNorm1DGPU';

  int numFeatures;
  double momentum;
  double epsilon;
  bool isTraining = true;

  /// Learnable scaling vector used to find the optimal scale for the normalized features.
  late GPUTensor<Vector> gamma;
  /// Learnable shifting vector (similar to bias) used to find the optimal offset.
  late GPUTensor<Vector> beta;
  /// Non-trainable parameter used to track the global average mean.
  late GPUTensor<Vector> runningMean;
  /// Non-trainable parameter used to track the global average variance.
  late GPUTensor<Vector> runningVariance;

  /// Persistent Cache for Static Unrolling
  int cacheBatchSize = -1;
  GPUTensor<Vector>? cachedOut;

  /// Requires the size of the input vector [numFeatures]. Initializes with a fixed [momentum] and a safety variance addition [epsilon] to prevent division by zero.
  BatchNorm1DTL(
      this.numFeatures, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      });

  /// Returns the trainable parameters [gamma] and [beta].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(gamma);
      params.add(beta);
    }
    return params;
  }

  /// Allocates VRAM for [gamma], [beta], [runningMean], and [runningVariance] in accordance with the given [numFeatures].
  @override
  void build(GPUTensor<Vector> input) {
    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];
    List<double> rmValues = <double>[];
    List<double> rvValues = <double>[];

    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(1.0);
      betaValues.add(0.0);
      rmValues.add(0.0);
      rvValues.add(1.0);
    }

    gamma = GPUTensor<Vector>(gammaValues);
    beta = GPUTensor<Vector>(betaValues);
    runningMean = GPUTensor<Vector>(rmValues);
    runningVariance = GPUTensor<Vector>(rvValues);

    built = true;
  }

  /// Writes the [batchNorm1dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// The [intermediates] list should be empty since it is not used in this operation.
  @override
  GPUTensor<Vector> forward(GPUTensor<Vector> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];
    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    GPUTensor<Vector> out = batchNorm1dGPU(
      input,
      gamma,
      beta,
      runningMean,
      runningVariance,
      momentum,
      epsilon,
      isTraining,
      tape,
      outTensor: cachedOut
    );
    return out;
  }

  /// Frees VRAM for [gamma], [beta], [runningMean], and [runningVariance].
  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }
  }

  /// Returns [gamma], [beta], [runningMean], and [runningVariance] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    gamma.toCpu();
    beta.toCpu();
    runningMean.toCpu();
    runningVariance.toCpu();

    wMap['gamma'] = gamma.value;
    wMap['beta'] = beta.value;
    wMap['runningMean'] = runningMean.value;
    wMap['runningVariance'] = runningVariance.value;

    return wMap;
  }

  /// Sets [gamma], [beta], [runningMean], and [runningVariance] using the provided map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }

    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];
    List<double> rmValues = <double>[];
    List<double> rvValues = <double>[];

    List<dynamic> rawGamma = newWeights['gamma']!;
    List<dynamic> rawBeta = newWeights['beta']!;
    List<dynamic> rawRM = newWeights['runningMean']!;
    List<dynamic> rawRV = newWeights['runningVariance']!;

    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(rawGamma[i] as double);
      betaValues.add(rawBeta[i] as double);
      rmValues.add(rawRM[i] as double);
      rvValues.add(rawRV[i] as double);
    }

    gamma = GPUTensor<Vector>(gammaValues);
    beta = GPUTensor<Vector>(betaValues);
    runningMean = GPUTensor<Vector>(rmValues);
    runningVariance = GPUTensor<Vector>(rvValues);

    built = true;
  }

  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }
}

/// Applies Batch Normalization over a 3D input [Tensor3D] to enforce a mean of 0 and a variance of 1.
/// It standardizes the spatial dimensions for each sub-element in the input [Tensor3D] (which are matrices) independently.
/// Usually, the different matrices building the [Tensor3D] can be viewed as different channels.
class BatchNorm2DTL extends TapeLayer<Tensor3D, Tensor3D> {
  String get name => 'BatchNorm2DGPU';

  int numChannels;
  double momentum;
  double epsilon;
  bool isTraining = true;

  /// Learnable scaling vector used to find the optimal scale for the normalized features per channel.
  late GPUTensor<Vector> gamma;
  /// Learnable shifting vector (similar to bias) used to find the optimal offset per channel.
  late GPUTensor<Vector> beta;
  /// Non-trainable parameter used to track the global average mean of the inputs per channel.
  late GPUTensor<Vector> runningMean;
  /// Non-trainable parameter used to track the global average variance of the inputs per channel.
  late GPUTensor<Vector> runningVariance;

  /// Persistent Cache for Static Unrolling
  int cacheBatchSize = -1;
  GPUTensor<Tensor3D>? cachedOut;

  /// Requires the number of channels [numChannels] (the amount of matrices in the input [Tensor3D]). Initializes with a fixed [momentum] and a safety variance addition [epsilon] to prevent division by zero.
  BatchNorm2DTL(
      this.numChannels, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      });

  /// Returns the trainable parameters [gamma] and [beta].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(gamma);
      params.add(beta);
    }
    return params;
  }

  /// Allocates VRAM for [gamma], [beta], [runningMean], and [runningVariance] in accordance with the given [numChannels].
  @override
  void build(GPUTensor<Tensor3D> input) {
    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];
    List<double> rmValues = <double>[];
    List<double> rvValues = <double>[];

    for (int i = 0; i < numChannels; i = i + 1) {
      gammaValues.add(1.0);
      betaValues.add(0.0);
      rmValues.add(0.0);
      rvValues.add(1.0);
    }

    gamma = GPUTensor<Vector>(gammaValues);
    beta = GPUTensor<Vector>(betaValues);
    runningMean = GPUTensor<Vector>(rmValues);
    runningVariance = GPUTensor<Vector>(rvValues);

    built = true;
  }

  /// Writes the [batchNorm2dGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// The [intermediates] list should be empty since it is not used in this operation.
  @override
  GPUTensor<Tensor3D> forward(GPUTensor<Tensor3D> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];
    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }
    GPUTensor<Tensor3D> out = batchNorm2dGPU(
      input,
      gamma,
      beta,
      runningMean,
      runningVariance,
      momentum,
      epsilon,
      isTraining,
      tape,
      outTensor: cachedOut
    );
    return out;
  }

  /// Frees VRAM for [gamma], [beta], [runningMean], and [runningVariance].
  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }
  }

  /// Returns [gamma], [beta], [runningMean], and [runningVariance] as a map.
  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    gamma.toCpu();
    beta.toCpu();
    runningMean.toCpu();
    runningVariance.toCpu();

    wMap['gamma'] = gamma.value;
    wMap['beta'] = beta.value;
    wMap['runningMean'] = runningMean.value;
    wMap['runningVariance'] = runningVariance.value;

    return wMap;
  }

  /// Sets [gamma], [beta], [runningMean], and [runningVariance] using the provided map.
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }

    List<double> gammaValues = <double>[];
    List<double> betaValues = <double>[];
    List<double> rmValues = <double>[];
    List<double> rvValues = <double>[];

    List<dynamic> rawGamma = newWeights['gamma']!;
    List<dynamic> rawBeta = newWeights['beta']!;
    List<dynamic> rawRM = newWeights['runningMean']!;
    List<dynamic> rawRV = newWeights['runningVariance']!;

    for (int i = 0; i < numChannels; i = i + 1) {
      gammaValues.add(rawGamma[i] as double);
      betaValues.add(rawBeta[i] as double);
      rmValues.add(rawRM[i] as double);
      rvValues.add(rawRV[i] as double);
    }

    gamma = GPUTensor<Vector>(gammaValues);
    beta = GPUTensor<Vector>(betaValues);
    runningMean = GPUTensor<Vector>(rmValues);
    runningVariance = GPUTensor<Vector>(rvValues);

    built = true;
  }

  @override
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }
}