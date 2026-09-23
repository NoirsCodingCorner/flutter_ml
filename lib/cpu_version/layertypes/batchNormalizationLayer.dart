
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class BatchNorm1D extends Layer<Vector, Vector> {
  @override
  String name = 'batch_norm_1d';
  int numFeatures;
  double momentum;
  double epsilon;
  bool isTraining = true;

  late Tensor<Vector> gamma;
  late Tensor<Vector> beta;
  late Vector runningMean;
  late Vector runningVariance;

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

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(gamma);
    params.add(beta);
    return params;
  }

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

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['gamma'] = gamma.value;
    weightsMap['beta'] = beta.value;
    weightsMap['runningMean'] = runningMean;
    weightsMap['runningVariance'] = runningVariance;
    return weightsMap;
  }

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

class BatchNorm2D extends Layer<Tensor3D, Tensor3D> {
  @override
  String name = 'batch_norm_2d';
  int numChannels;
  double momentum;
  double epsilon;
  bool isTraining = true;

  late Tensor<Vector> gamma;
  late Tensor<Vector> beta;
  late Vector runningMean;
  late Vector runningVariance;

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

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(gamma);
    params.add(beta);
    return params;
  }

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

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['gamma'] = gamma.value;
    weightsMap['beta'] = beta.value;
    weightsMap['runningMean'] = runningMean;
    weightsMap['runningVariance'] = runningVariance;
    return weightsMap;
  }

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
