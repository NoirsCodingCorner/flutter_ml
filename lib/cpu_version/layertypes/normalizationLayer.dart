import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class LayerNormalization extends Layer<Matrix, Matrix> {
  @override
  String name = 'layer_norm';
  double epsilon;

  late Tensor<Vector> gamma;
  late Tensor<Vector> beta;

  LayerNormalization({this.epsilon = 1e-5});

  @override
  List<Tensor> get parameters => [gamma, beta];

  @override
  void build(Tensor<Matrix> input) {
    int numFeatures = input.shape[1];

    List<double> gammaValues = [];
    List<double> betaValues = [];
    for (int i = 0; i < numFeatures; i = i + 1) {
      gammaValues.add(1.0);
      betaValues.add(0.0);
    }

    gamma = Tensor<Vector>(gammaValues);
    gamma.shape = [numFeatures];
    beta = Tensor<Vector>(betaValues);
    beta.shape = [numFeatures];

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    int numRows = input.shape[0];
    int numCols = input.shape[1];

    List<double> outValue = [];
    List<double> normalizedData = []; // Store for backward
    List<double> means = [];
    List<double> variances = [];

    for (int r = 0; r < numRows; r = r + 1) {
      int offset = r * numCols;
      double sum = 0;
      for (int c = 0; c < numCols; c = c + 1) {
        sum = sum + input.data[offset + c];
      }
      double mean = sum / numCols;
      means.add(mean);

      double varSum = 0;
      for (int c = 0; c < numCols; c = c + 1) {
        varSum = varSum + (input.data[offset + c] - mean) * (input.data[offset + c] - mean);
      }
      double variance = varSum / numCols;
      variances.add(variance);

      double invStd = 1.0 / sqrt(variance + epsilon);
      for (int c = 0; c < numCols; c = c + 1) {
        double norm = (input.data[offset + c] - mean) * invStd;
        normalizedData.add(norm);
        outValue.add(gamma.data[c] * norm + beta.data[c]);
      }
    }

    Tensor<Matrix> out = Tensor<Matrix>(outValue);
    out.shape = [numRows, numCols];

    out.creator = Node([input, gamma, beta], () {
      for (int r = 0; r < numRows; r = r + 1) {
        int offset = r * numCols;
        double sumGradXHat = 0;
        double dotProductTerm = 0;

        List<double> gradXHatRow = [];
        for (int c = 0; c < numCols; c = c + 1) {
          int idx = offset + c;
          double gXHat = out.grad[idx] * gamma.data[c];
          gradXHatRow.add(gXHat);
          sumGradXHat = sumGradXHat + gXHat;
          dotProductTerm = dotProductTerm + gXHat * normalizedData[idx];

          gamma.grad[c] = gamma.grad[c] + out.grad[idx] * normalizedData[idx];
          beta.grad[c] = beta.grad[c] + out.grad[idx];
        }

        double invStd = 1.0 / (numCols * sqrt(variances[r] + epsilon));
        for (int c = 0; c < numCols; c = c + 1) {
          int idx = offset + c;
          double totalGrad = invStd * (numCols * gradXHatRow[c] - sumGradXHat - normalizedData[idx] * dotProductTerm);
          input.grad[idx] = input.grad[idx] + totalGrad;
        }
      }
    }, opName: 'layer_norm');

    return out;
  }

  @override
  Map<String, dynamic> getWeights() => {'gamma': gamma.value, 'beta': beta.value};

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<double> newGamma = (weightsMap['gamma'] as List).cast<double>();
    List<double> newBeta = (weightsMap['beta'] as List).cast<double>();
    for (int i = 0; i < gamma.data.length; i = i + 1) {
      gamma.data[i] = newGamma[i];
      beta.data[i] = newBeta[i];
    }
  }
}