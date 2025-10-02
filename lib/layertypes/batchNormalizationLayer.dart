import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A 1D Batch Normalization layer.
///
/// This layer normalizes its inputs to have a mean of approximately 0 and a
/// standard deviation of approximately 1. This helps to stabilize and accelerate
/// the training of deep neural networks.
///
/// It maintains a running average of the mean and variance of the data it sees
/// during training. It also has two trainable parameters: `gamma` (a scaling
/// factor) and `beta` (a shifting factor).
///
/// **IMPORTANT:** This layer behaves differently during training and inference.
/// You must manually set the `isTraining` flag to `false` before evaluating
/// your model.
///
/// - **Input:** A `Tensor<Vector>` of shape `[num_features]`.
/// - **Output:** A `Tensor<Vector>` of the same shape.
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

    for (int i = 0; i < numFeatures; i++) {
      xHat[i] = (x.value[i] - currentMean[i]) / sqrt(currentVariance[i] + epsilon);
    }

    Vector outValue = [];
    for (int i = 0; i < numFeatures; i++) {
      outValue.add(gamma.value[i] * xHat[i] + beta.value[i]);
    }

    Tensor<Vector> out = Tensor<Vector>(outValue);
    out.creator = Node([x, gamma, beta], () {
      for(int i=0; i < numFeatures; i++){
        double invStd = 1 / sqrt(currentVariance[i] + epsilon);
        gamma.grad[i] += out.grad[i] * xHat[i];
        beta.grad[i] += out.grad[i];
        x.grad[i] += out.grad[i] * gamma.value[i] * invStd;
      }
    }, opName: 'batch_norm_1d');

    return out;
  }
}

/// A 2D Batch Normalization layer.
///
/// This layer normalizes its inputs across the spatial dimensions (`height`, `width`)
/// for each channel independently. It is designed to be used after a `Conv2DLayer`
/// and before its activation function.
///
/// It maintains a running average of the mean and variance for each channel and
/// has trainable `gamma` (scale) and `beta` (shift) parameters for each channel.
///
/// **IMPORTANT:** Set the `isTraining` flag to `false` during evaluation.
///
/// - **Input:** A `Tensor<Tensor3D>` of shape `[channels, height, width]`.
/// - **Output:** A `Tensor<Tensor3D>` of the same shape.
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

    // Calculate mean and variance for each channel
    double numElements = (height * width).toDouble();
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

    Vector meanToUse;
    Vector varianceToUse;

    if (isTraining) {
      for (int c = 0; c < numChannels; c++) {
        runningMean[c] = momentum * runningMean[c] + (1 - momentum) * currentMean[c];
        runningVariance[c] = momentum * runningVariance[c] + (1 - momentum) * currentVariance[c];
      }
      meanToUse = runningMean;
      varianceToUse = runningVariance;
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
      // Backward pass (simplified, treats mean/variance as constants)
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
}