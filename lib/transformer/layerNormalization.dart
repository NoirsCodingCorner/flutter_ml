import 'dart:math';

import 'package:flutter_ml/layertypes/reluLayer.dart';

import '../activationFunctions/relu.dart';
import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import '../layertypes/batchNormalizationLayer.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/dropout.dart';
import '../layertypes/layer.dart';

/// A Layer Normalization layer for 1D data (Vectors).
///
/// This layer normalizes its inputs across the feature dimension for a single
/// data sample. It is designed to work on the vector outputs of layers like
/// `DenseLayer` or `RNN`.
///
/// - **Input:** A `Tensor<Vector>` of shape `[num_features]`.
/// - **Output:** A `Tensor<Vector>` of the same shape.
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
    return layerNorm1D(input as Tensor<Vector>, gamma, beta, epsilon: epsilon);
  }
}
Tensor<Vector> layerNorm1D(Tensor<Vector> v, Tensor<Vector> gamma, Tensor<Vector> beta, {double epsilon = 1e-5}) {
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
  for (int i = 0; i < numFeatures; i++) {
    outValue.add(gamma.value[i] * normalizedVector[i] + beta.value[i]);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);
  int cost = numFeatures * 8; // Approximate cost

  out.creator = Node([v, gamma, beta], () {
    double stdInv = 1.0 / sqrt(variance + epsilon);

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
  }, opName: 'layer_norm_1d', cost: cost);
  return out;
}


void evaluateModel(SNetwork network, List<Vector> inputs, List<Vector> targets, String setName) {
  int correctPredictions = 0;
  int numSamples = inputs.length;

  // --- Ensure network is in INFERENCE mode ---
  for (Layer layer in network.layers) {
    if (layer is BatchNorm1D) {
      BatchNorm1D batchNormLayer = layer as BatchNorm1D;
      batchNormLayer.isTraining = false;
    }
    // Set DropoutLayer to NOT training
    if (layer is DropoutLayer) {
      DropoutLayer dropoutLayer = layer as DropoutLayer;
      dropoutLayer.isTraining = false;
    }
  }

  for (int i = 0; i < numSamples; i++) {
    Tensor<Vector> testInput = Tensor<Vector>(inputs[i]);
    Tensor<Vector> pred = network.predict(testInput) as Tensor<Vector>;
    int result = (pred.value[0] > 0.5) ? 1 : 0;
    if (result == targets[i][0]) {
      correctPredictions++;
    }
  }

  double accuracy = (correctPredictions / numSamples) * 100.0;
  print('${setName} Accuracy: ${accuracy.toStringAsFixed(2)}%');
}

// --- HELPER FUNCTION FOR TRAINING AND EVALUATION ---
void trainAndEvaluate(SNetwork network, List<Vector> trainInputs, List<Vector> trainTargets, List<Vector> testInputs, List<Vector> testTargets) {
  int numTrainSamples = trainInputs.length;
  int epochs = 200;

  network.predict(Tensor<Vector>(trainInputs[0]));

  network.compile(
      configuredOptimizer: Adam(network.parameters, learningRate: 0.005)
  );

  Optimizer optimizer = network.optimizer;

  // --- Ensure network is in TRAINING mode ---
  for (Layer layer in network.layers) {
    if (layer is BatchNorm1D) {
      BatchNorm1D batchNormLayer = layer as BatchNorm1D;
      batchNormLayer.isTraining = true;
    }
    // Set DropoutLayer to TRAINING
    if (layer is DropoutLayer) {
      DropoutLayer dropoutLayer = layer as DropoutLayer;
      dropoutLayer.isTraining = true;
    }
  }

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < numTrainSamples; i++) {
      optimizer.zeroGrad();

      Tensor<Vector> inputTensor = Tensor<Vector>(trainInputs[i]);
      Tensor<Vector> targetTensor = Tensor<Vector>(trainTargets[i]);

      Tensor<Vector> finalOutput = network.predict(inputTensor) as Tensor<Vector>;
      Tensor<Scalar> loss = mse(finalOutput, targetTensor);

      loss.backward();
      optimizer.step();
    }
  }

  // --- Evaluation ---
  print('Training Complete.');
  // 1. Check for memorization/overfitting (Training Set)
  evaluateModel(network, trainInputs, trainTargets, '  Training Set');
  // 2. Check for generalization (Test Set - the true metric)
  evaluateModel(network, testInputs, testTargets, '  Test Set');
}

void main() {
  // --- 1. PREPARE THE DATASET ---
  int numTotalSamples = 1000;
  List<Vector> allInputs = <Vector>[];
  List<Vector> allTargets = <Vector>[];
  Random random = Random();

  for (int i = 0; i < numTotalSamples; i++) {
    double x = (random.nextDouble() * 4.0) - 2.0;
    double y = (random.nextDouble() * 4.0) - 2.0;
    allInputs.add(<double>[x, y]);
    double distance = sqrt(x * x + y * y);
    allTargets.add(<double>[(distance < 1.5) ? 1.0 : 0.0]);
  }

  // --- 2. SPLIT THE DATASET (80% Train, 20% Test) ---
  int splitIndex = (numTotalSamples * 0.8).toInt();
  List<Vector> trainInputs = <Vector>[];
  List<Vector> trainTargets = <Vector>[];
  List<Vector> testInputs = <Vector>[];
  List<Vector> testTargets = <Vector>[];

  for (int i = 0; i < numTotalSamples; i++) {
    if (i < splitIndex) {
      trainInputs.add(allInputs[i]);
      trainTargets.add(allTargets[i]);
    } else {
      testInputs.add(allInputs[i]);
      testTargets.add(allTargets[i]);
    }
  }

  // --- 3. DEFINE THE MODELS with Dropout ---
  double dropoutRate = 0.2;

  // Model A: Plain Deep Network + Dropout
  SNetwork plainDeepNetwork = SNetwork(<Layer>[
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(16, activation: ReLU()),
    DropoutLayer(dropoutRate),
    DenseLayer(1, activation: Sigmoid()),
  ]);

  // Model B: Normalized Deep Network + Dropout
  SNetwork normalizedDeepNetwork = SNetwork(<Layer>[
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(16),
    LayerNormalization(),
    ReLULayer(),
    DropoutLayer(dropoutRate),
    DenseLayer(1, activation: Sigmoid()),
  ]);

  // --- 4. RUN THE EXPERIMENTS ---
  print('--- Training Plain Deep Network with Dropout (EXPECTED TO STRUGGLE) ---');
  trainAndEvaluate(plainDeepNetwork, trainInputs, trainTargets, testInputs, testTargets);

  print('\n' + '='*50 + '\n');

  print('--- Training Deep Network with Layer Normalization and Dropout (EXPECTED TO SUCCEED) ---');
  trainAndEvaluate(normalizedDeepNetwork, trainInputs, trainTargets, testInputs, testTargets);
}