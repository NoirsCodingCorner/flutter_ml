import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

class BatchNorm1D extends Layer {
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
    for (int i = 0; i < numFeatures; i++) {
      gammaValues.add(1.0);
    }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for (int i = 0; i < numFeatures; i++) {
      betaValues.add(0.0);
    }
    beta = Tensor<Vector>(betaValues);

    runningMean = [];
    for (int i = 0; i < numFeatures; i++) {
      runningMean.add(0.0);
    }

    runningVariance = [];
    for (int i = 0; i < numFeatures; i++) {
      runningVariance.add(1.0);
    }
  }

  @override
  List<Tensor> get parameters => [gamma, beta];

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Tensor<Vector> x = input as Tensor<Vector>;
    Vector xHat = [];
    for (int i = 0; i < numFeatures; i++) {
      xHat.add(0.0);
    }

    Vector currentMean;
    Vector currentVariance;

    if (isTraining) {
      currentMean = x.value;
      currentVariance = [];
      for (int i = 0; i < numFeatures; i++) {
        currentVariance.add(0.0);
      }

      for (int i = 0; i < numFeatures; i++) {
        runningMean[i] = momentum * runningMean[i] + (1 - momentum) * currentMean[i];
        runningVariance[i] = momentum * runningVariance[i] + (1 - momentum) * currentVariance[i];
      }
    } else {
      currentMean = runningMean;
      currentVariance = runningVariance;
    }

    Vector varianceToUse = isTraining ? runningVariance : currentVariance;

    for (int i = 0; i < numFeatures; i++) {
      double meanToUse = isTraining ? runningMean[i] : currentMean[i];
      xHat[i] = (x.value[i] - meanToUse) / sqrt(varianceToUse[i] + epsilon);
    }

    Vector outValue = [];
    for (int i = 0; i < numFeatures; i++) {
      outValue.add(gamma.value[i] * xHat[i] + beta.value[i]);
    }

    Tensor<Vector> out = Tensor<Vector>(outValue);
    out.creator = Node([x, gamma, beta], () {
      for(int i=0; i < numFeatures; i++){
        double invStd = 1 / sqrt(varianceToUse[i] + epsilon);
        gamma.grad[i] += out.grad[i] * xHat[i];
        beta.grad[i] += out.grad[i];
        x.grad[i] += out.grad[i] * gamma.value[i] * invStd;
      }
    }, opName: 'batch_norm_1d');

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'gamma': gamma.value,
      'beta': beta.value,
      'runningMean': runningMean,
      'runningVariance': runningVariance,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Vector newGamma = (weightsMap['gamma'] as List).map((dynamic e) => e as double).toList();
    Vector newBeta = (weightsMap['beta'] as List).map((dynamic e) => e as double).toList();
    Vector newRunningMean = (weightsMap['runningMean'] as List).map((dynamic e) => e as double).toList();
    Vector newRunningVariance = (weightsMap['runningVariance'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < numFeatures; i++) {
      gamma.value[i] = newGamma[i];
      beta.value[i] = newBeta[i];
      runningMean[i] = newRunningMean[i];
      runningVariance[i] = newRunningVariance[i];
    }
  }
}

class BatchNorm2D extends Layer {
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
    for (int i = 0; i < numChannels; i++) { gammaValues.add(1.0); }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for (int i = 0; i < numChannels; i++) { betaValues.add(0.0); }
    beta = Tensor<Vector>(betaValues);

    runningMean = [];
    for (int i = 0; i < numChannels; i++) { runningMean.add(0.0); }

    runningVariance = [];
    for (int i = 0; i < numChannels; i++) { runningVariance.add(1.0); }
  }

  @override
  List<Tensor> get parameters => [gamma, beta];

  @override
  Tensor<Tensor3D> forward(Tensor<dynamic> input) {
    Tensor<Tensor3D> x = input as Tensor<Tensor3D>;
    int height = x.value[0].length;
    int width = x.value[0][0].length;

    Tensor3D xHat = [];
    for (int c = 0; c < numChannels; c++) {
      Matrix m = [];
      for (int h = 0; h < height; h++) {
        m.add(List<double>.filled(width, 0.0));
      }
      xHat.add(m);
    }

    Vector currentMean = List<double>.filled(numChannels, 0.0);
    Vector currentVariance = List<double>.filled(numChannels, 0.0);
    Vector meanToUse;
    Vector varianceToUse;

    double numElements = (height * width).toDouble();

    if (isTraining) {
      for (int c = 0; c < numChannels; c++) {
        double sum = 0;
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            sum += x.value[c][h][w];
          }
        }
        currentMean[c] = sum / numElements;

        double varianceSum = 0;
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            varianceSum += pow(x.value[c][h][w] - currentMean[c], 2);
          }
        }
        currentVariance[c] = varianceSum / numElements;
      }

      for (int c = 0; c < numChannels; c++) {
        runningMean[c] = momentum * runningMean[c] + (1 - momentum) * currentMean[c];
        runningVariance[c] = momentum * runningVariance[c] + (1 - momentum) * currentVariance[c];
      }
      meanToUse = currentMean;
      varianceToUse = currentVariance;
    } else {
      meanToUse = runningMean;
      varianceToUse = runningVariance;
    }

    for (int c = 0; c < numChannels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          xHat[c][h][w] = (x.value[c][h][w] - meanToUse[c]) / sqrt(varianceToUse[c] + epsilon);
        }
      }
    }

    Tensor3D outValue = [];
    for (int c = 0; c < numChannels; c++) {
      Matrix m = [];
      for (int h = 0; h < height; h++) {
        Vector row = [];
        for (int w = 0; w < width; w++) {
          row.add(gamma.value[c] * xHat[c][h][w] + beta.value[c]);
        }
        m.add(row);
      }
      outValue.add(m);
    }

    Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
    out.creator = Node([x, gamma, beta], () {
      for (int c = 0; c < numChannels; c++) {
        double invStd = 1 / sqrt(varianceToUse[c] + epsilon);
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            gamma.grad[c] += out.grad[c][h][w] * xHat[c][h][w];
            beta.grad[c] += out.grad[c][h][w];
            x.grad[c][h][w] += out.grad[c][h][w] * gamma.value[c] * invStd;
          }
        }
      }
    }, opName: 'batch_norm_2d');

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'gamma': gamma.value,
      'beta': beta.value,
      'runningMean': runningMean,
      'runningVariance': runningVariance,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Vector newGamma = (weightsMap['gamma'] as List).map((dynamic e) => e as double).toList();
    Vector newBeta = (weightsMap['beta'] as List).map((dynamic e) => e as double).toList();
    Vector newRunningMean = (weightsMap['runningMean'] as List).map((dynamic e) => e as double).toList();
    Vector newRunningVariance = (weightsMap['runningVariance'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < numChannels; i++) {
      gamma.value[i] = newGamma[i];
      beta.value[i] = newBeta[i];
      runningMean[i] = newRunningMean[i];
      runningVariance[i] = newRunningVariance[i];
    }
  }
}