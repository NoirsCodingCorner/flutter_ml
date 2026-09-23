import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class BatchNorm1DTL extends TapeLayer {
  int numFeatures;
  double momentum;
  double epsilon;
  bool isTraining = true;

  late GPUTensor<Vector> gamma;
  late GPUTensor<Vector> beta;
  late GPUTensor<Vector> runningMean;
  late GPUTensor<Vector> runningVariance;

  BatchNorm1DTL(
      this.numFeatures, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      });

  @override
  String get name {
    return 'BatchNorm1DGPU';
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

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Vector> typedInput = input as GPUTensor<Vector>;

    GPUTensor<Vector> out = batchNorm1dGPU(
      typedInput,
      gamma,
      beta,
      runningMean,
      runningVariance,
      momentum,
      epsilon,
      isTraining,
      tape,
    );

    return out;
  }

  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }
  }

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
}

class BatchNorm2DTL extends TapeLayer {
  int numChannels;
  double momentum;
  double epsilon;
  bool isTraining = true;

  late GPUTensor<Vector> gamma;
  late GPUTensor<Vector> beta;
  late GPUTensor<Vector> runningMean;
  late GPUTensor<Vector> runningVariance;

  BatchNorm2DTL(
      this.numChannels, {
        this.momentum = 0.9,
        this.epsilon = 1e-5,
      });

  @override
  String get name {
    return 'BatchNorm2DGPU';
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

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Tensor3D> typedInput = input as GPUTensor<Tensor3D>;

    GPUTensor<Tensor3D> out = batchNorm2dGPU(
      typedInput,
      gamma,
      beta,
      runningMean,
      runningVariance,
      momentum,
      epsilon,
      isTraining,
      tape,
    );

    return out;
  }

  @override
  void free() {
    if (built) {
      gamma.free();
      beta.free();
      runningMean.free();
      runningVariance.free();
    }
  }

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
}