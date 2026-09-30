import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'layer.dart';

/// Applies Batch Normalization over a 1D [Vector] tensor.
/// Normalizes the input features across the batch to have zero mean and unit variance,
/// followed by a learnable affine transformation using [gamma] and [beta].
/// Tracks running statistics [runningMean] and [runningVariance] for evaluation during inference.
class BatchNorm1D extends Layer<Vector, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'batch_norm_1d';

  /// The number of feature dimensions expected in the input vector.
  int numFeatures;

  /// The momentum factor used for the running mean and variance computation.
  double momentum;

  /// A small value added to the denominator for numerical stability to prevent division by zero.
  double epsilon;

  /// Bool flag indicating whether the layer is in training mode or inference mode.
  bool isTraining = true;

  /// Learnable scale parameter tensor initialized to 1.0.
  late Tensor<Vector> gamma;

  /// Learnable shift parameter tensor initialized to 0.0.
  late Tensor<Vector> beta;

  /// Running mean vector tracked during training for inference.
  late Vector runningMean;

  /// Running variance vector tracked during training for inference.
  late Vector runningVariance;

  /// Creates a [BatchNorm1D] layer with [numFeatures], [momentum], and [epsilon].
  /// Allocates and initializes [gamma] with 1.0, [beta] with 0.0, [runningMean] with 0.0, and [runningVariance] with 1.0.
  BatchNorm1D(
      this.numFeatures, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      }) {
    Vector gammaValues = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(1.0);
    }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      betaValues.add(0.0);
    }
    beta = Tensor<Vector>(betaValues);

    runningMean = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      runningMean.add(0.0);
    }

    runningVariance = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      runningVariance.add(1.0);
    }
  }

  /// Returns the trainable parameters [gamma] and [beta].
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(gamma);
    params.add(beta);
    return params;
  }

  /// Executes the 1D batch normalization forward pass on the CPU using [batchNorm1dMath].
  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    return batchNorm1dMath(
      input,
      gamma,
      beta,
      runningMean,
      runningVariance,
      numFeatures,
      isTraining,
      momentum,
      epsilon,
    );
  }

  /// Returns [gamma], [beta], [runningMean], and [runningVariance] as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['gamma'] = gamma.value;
    weightsMap['beta'] = beta.value;
    weightsMap['runningMean'] = runningMean;
    weightsMap['runningVariance'] = runningVariance;
    return weightsMap;
  }

  /// Sets [gamma], [beta], [runningMean], and [runningVariance] from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> gammaDynamic = weightsMap['gamma'] as List<dynamic>;
    List<dynamic> betaDynamic = weightsMap['beta'] as List<dynamic>;
    List<dynamic> rmDynamic = weightsMap['runningMean'] as List<dynamic>;
    List<dynamic> rvDynamic = weightsMap['runningVariance'] as List<dynamic>;

    Vector newGamma = [];
    Vector newBeta = [];
    Vector newRunningMean = [];
    Vector newRunningVariance = [];

    for (int i = 0; i < numFeatures; i = i + 1) {
      newGamma.add(gammaDynamic[i] as double);
      newBeta.add(betaDynamic[i] as double);
      newRunningMean.add(rmDynamic[i] as double);
      newRunningVariance.add(rvDynamic[i] as double);
    }

    for (int i = 0; i < numFeatures; i = i + 1) {
      gamma.value[i] = newGamma[i];
      beta.value[i] = newBeta[i];
      runningMean[i] = newRunningMean[i];
      runningVariance[i] = newRunningVariance[i];
    }
  }
}

/// Applies Batch Normalization over a 3D [Tensor3D] tensor across channels.
/// Normalizes the channel dimensions across spatial elements to have zero mean and unit variance,
/// followed by a learnable affine transformation using [gamma] and [beta].
/// Tracks running statistics [runningMean] and [runningVariance] for evaluation during inference.
class BatchNorm2D extends Layer<Tensor3D, Tensor3D> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'batch_norm_2d';

  /// The number of channel dimensions expected in the input 3D tensor.
  int numChannels;

  /// The momentum factor used for the running mean and variance computation.
  double momentum;

  /// A small value added to the denominator for numerical stability to prevent division by zero.
  double epsilon;

  /// Bool flag indicating whether the layer is in training mode or inference mode.
  bool isTraining = true;

  /// Learnable scale parameter vector tensor across channels initialized to 1.0.
  late Tensor<Vector> gamma;

  /// Learnable shift parameter vector tensor across channels initialized to 0.0.
  late Tensor<Vector> beta;

  /// Running mean vector across channels tracked during training for inference.
  late Vector runningMean;

  /// Running variance vector across channels tracked during training for inference.
  late Vector runningVariance;

  /// Creates a [BatchNorm2D] layer with [numChannels], [momentum], and [epsilon].
  /// Allocates and initializes [gamma] with 1.0, [beta] with 0.0, [runningMean] with 0.0, and [runningVariance] with 1.0.
  BatchNorm2D(
      this.numChannels, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      }) {
    Vector gammaValues = [];
    for (int i = 0; i < numChannels; i = i + 1) {
      gammaValues.add(1.0);
    }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for (int i = 0; i < numChannels; i = i + 1) {
      betaValues.add(0.0);
    }
    beta = Tensor<Vector>(betaValues);

    runningMean = [];
    for (int i = 0; i < numChannels; i = i + 1) {
      runningMean.add(0.0);
    }

    runningVariance = [];
    for (int i = 0; i < numChannels; i = i + 1) {
      runningVariance.add(1.0);
    }
  }

  /// Returns the trainable parameters [gamma] and [beta].
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(gamma);
    params.add(beta);
    return params;
  }

  /// Executes the 2D batch normalization forward pass on the CPU using [batchNorm2dMath].
  @override
  Tensor<Tensor3D> forward(Tensor<Tensor3D> input) {
    return batchNorm2dMath(
      input,
      gamma,
      beta,
      runningMean,
      runningVariance,
      numChannels,
      isTraining,
      momentum,
      epsilon,
    );
  }

  /// Returns [gamma], [beta], [runningMean], and [runningVariance] as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['gamma'] = gamma.value;
    weightsMap['beta'] = beta.value;
    weightsMap['runningMean'] = runningMean;
    weightsMap['runningVariance'] = runningVariance;
    return weightsMap;
  }

  /// Sets [gamma], [beta], [runningMean], and [runningVariance] from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> gammaDynamic = weightsMap['gamma'] as List<dynamic>;
    List<dynamic> betaDynamic = weightsMap['beta'] as List<dynamic>;
    List<dynamic> rmDynamic = weightsMap['runningMean'] as List<dynamic>;
    List<dynamic> rvDynamic = weightsMap['runningVariance'] as List<dynamic>;

    Vector newGamma = [];
    Vector newBeta = [];
    Vector newRunningMean = [];
    Vector newRunningVariance = [];

    for (int i = 0; i < numChannels; i = i + 1) {
      newGamma.add(gammaDynamic[i] as double);
      newBeta.add(betaDynamic[i] as double);
      newRunningMean.add(rmDynamic[i] as double);
      newRunningVariance.add(rvDynamic[i] as double);
    }

    for (int i = 0; i < numChannels; i = i + 1) {
      gamma.value[i] = newGamma[i];
      beta.value[i] = newBeta[i];
      runningMean[i] = newRunningMean[i];
      runningVariance[i] = newRunningVariance[i];
    }
  }
}