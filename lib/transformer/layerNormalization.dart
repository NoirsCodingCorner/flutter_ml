import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';

class LayerNormalization extends Layer {
  @override
  String name = 'layer_norm';
  double epsilon;

  late Tensor<Vector> gamma;
  late Tensor<Vector> beta;

  LayerNormalization({this.epsilon = 1e-5});

  @override
  List<Tensor> get parameters => [gamma, beta];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int numFeatures = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;

    Vector gammaValues = [];
    for(int i=0; i<numFeatures; i++){ gammaValues.add(1.0); }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for(int i=0; i<numFeatures; i++){ betaValues.add(0.0); }
    beta = Tensor<Vector>(betaValues);

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    return layerNorm(input as Tensor<Matrix>, gamma, beta, epsilon: epsilon);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'gamma': gamma.value,
      'beta': beta.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Vector newGamma = (weightsMap['gamma'] as List).map((dynamic e) => e as double).toList();
    Vector newBeta = (weightsMap['beta'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < gamma.value.length; i++) {
      gamma.value[i] = newGamma[i];
    }
    for (int i = 0; i < beta.value.length; i++) {
      beta.value[i] = newBeta[i];
    }
  }
}

class LayerNormalizationVector extends Layer {
  @override
  String name = 'layer_norm_vector';
  double epsilon;

  late Tensor<Vector> gamma;
  late Tensor<Vector> beta;

  LayerNormalizationVector({this.epsilon = 1e-5});

  @override
  List<Tensor> get parameters => [gamma, beta];

  @override
  void build(Tensor<dynamic> input) {
    Vector inputValue = input.value as Vector;
    int numFeatures = inputValue.length;

    Vector gammaValues = [];
    for(int i=0; i<numFeatures; i++){ gammaValues.add(1.0); }
    gamma = Tensor<Vector>(gammaValues);

    Vector betaValues = [];
    for(int i=0; i<numFeatures; i++){ betaValues.add(0.0); }
    beta = Tensor<Vector>(betaValues);

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    return layerNormVector(input as Tensor<Vector>, gamma, beta, epsilon: epsilon);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'gamma': gamma.value,
      'beta': beta.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    Vector newGamma = (weightsMap['gamma'] as List).map((dynamic e) => e as double).toList();
    Vector newBeta = (weightsMap['beta'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < gamma.value.length; i++) {
      gamma.value[i] = newGamma[i];
    }
    for (int i = 0; i < beta.value.length; i++) {
      beta.value[i] = newBeta[i];
    }
  }
}

Tensor<Vector> layerNormVector(Tensor<Vector> v, Tensor<Vector> gamma, Tensor<Vector> beta, {double epsilon = 1e-5}) {
  int numFeatures = v.value.length;

  double mean = 0;
  for (double val in v.value) { mean += val; }
  mean /= numFeatures;

  double variance = 0;
  for (double val in v.value) { variance += pow(val - mean, 2); }
  variance /= numFeatures;

  Vector normalizedVector = [];
  for (double val in v.value) {
    normalizedVector.add((val - mean) / sqrt(variance + epsilon));
  }

  Vector outValue = [];
  for (int c = 0; c < numFeatures; c++) {
    outValue.add(gamma.value[c] * normalizedVector[c] + beta.value[c]);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);
  int cost = numFeatures * 8;

  out.creator = Node([v, gamma, beta], () {
    Vector grad_x_hat = [];
    for(int c=0; c < numFeatures; c++){
      grad_x_hat.add(out.grad[c] * gamma.value[c]);
      gamma.grad[c] += out.grad[c] * normalizedVector[c];
      beta.grad[c] += out.grad[c];
    }

    double sum_grad_x_hat = 0;
    for (double val in grad_x_hat) { sum_grad_x_hat += val; }

    double dot_product_term = 0;
    for (int c = 0; c < numFeatures; c++) {
      dot_product_term += grad_x_hat[c] * normalizedVector[c];
    }

    for (int c = 0; c < numFeatures; c++) {
      double term1 = numFeatures * grad_x_hat[c];
      double term2 = sum_grad_x_hat;
      double term3 = normalizedVector[c] * dot_product_term;

      double total_grad = (1.0 / (numFeatures * sqrt(variance + epsilon))) * (term1 - term2 - term3);
      v.grad[c] += total_grad;
    }
  }, opName: 'layer_norm_vector', cost: cost);
  return out;
}

Tensor<Matrix> layerNorm(Tensor<Matrix> m, Tensor<Vector> gamma, Tensor<Vector> beta, {double epsilon = 1e-5}) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  Matrix normalizedRows = [];
  Vector means = [];
  Vector variances = [];

  for (int r = 0; r < numRows; r++) {
    Vector row = m.value[r];
    double mean = 0;
    for (double val in row) { mean += val; }
    mean /= numCols;
    means.add(mean);

    double variance = 0;
    for (double val in row) { variance += pow(val - mean, 2); }
    variance /= numCols;
    variances.add(variance);

    Vector normalizedRow = [];
    for (double val in row) {
      normalizedRow.add((val - mean) / sqrt(variance + epsilon));
    }
    normalizedRows.add(normalizedRow);

    Vector finalRow = [];
    for (int c = 0; c < numCols; c++) {
      finalRow.add(gamma.value[c] * normalizedRow[c] + beta.value[c]);
    }
    outValue.add(finalRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = numRows * numCols * 8;

  out.creator = Node([m, gamma, beta], () {
    for(int r = 0; r < numRows; r++){
      Vector grad_x_hat = [];
      for(int c=0; c < numCols; c++){
        grad_x_hat.add(out.grad[r][c] * gamma.value[c]);
        gamma.grad[c] += out.grad[r][c] * normalizedRows[r][c];
        beta.grad[c] += out.grad[r][c];
      }

      double sum_grad_x_hat = 0;
      for (double val in grad_x_hat) { sum_grad_x_hat += val; }

      double dot_product_term = 0;
      for (int c = 0; c < numCols; c++) {
        dot_product_term += grad_x_hat[c] * normalizedRows[r][c];
      }

      for (int c = 0; c < numCols; c++) {
        double term1 = numCols * grad_x_hat[c];
        double term2 = sum_grad_x_hat;
        double term3 = normalizedRows[r][c] * dot_product_term;

        double total_grad = (1.0 / (numCols * sqrt(variances[r] + epsilon))) * (term1 - term2 - term3);
        m.grad[r][c] += total_grad;
      }
    }
  }, opName: 'layer_norm', cost: cost);
  return out;
}