import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import '../optimizers/sgd.dart';
import 'denseLayer.dart';
import 'layer.dart';
import 'lstmLayer.dart';

class MultiTierLSTMLayer extends Layer {
  @override
  String name = 'multitier_lstm';

  final int hiddenSize;
  final List<int> tierClockCycles;
  final int numTiers;

  late List<Tensor<Matrix>> W_f_tiers, W_i_tiers, W_c_tiers, W_o_tiers;
  late List<Tensor<Vector>> b_f_tiers, b_i_tiers, b_c_tiers, b_o_tiers;

  late List<int> cumulativeClockCycles;

  MultiTierLSTMLayer(this.hiddenSize, {required this.tierClockCycles})
      : numTiers = tierClockCycles.length + 1 {
    cumulativeClockCycles = [];
    int product = 1;
    for (int cycle in tierClockCycles) {
      product *= cycle;
      cumulativeClockCycles.add(product);
    }
  }

  @override
  List<Tensor> get parameters {
    List<Tensor> allParams = [];
    for (int i = 0; i < numTiers; i++) {
      allParams.addAll([
        W_f_tiers[i], b_f_tiers[i], W_i_tiers[i], b_i_tiers[i],
        W_c_tiers[i], b_c_tiers[i], W_o_tiers[i], b_o_tiers[i]
      ]);
    }
    return allParams;
  }

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    W_f_tiers = []; W_i_tiers = []; W_c_tiers = []; W_o_tiers = [];
    b_f_tiers = []; b_i_tiers = []; b_c_tiers = []; b_o_tiers = [];

    Tensor<Matrix> initWeights(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      Matrix values = [];
      for (int i = 0; i < fanOut; i++) {
        Vector row = [];
        for (int j = 0; j < fanIn; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    for (int i = 0; i < numTiers; i++) {
      int combinedSize;
      if (i == 0) {
        combinedSize = hiddenSize + ((numTiers - 1) * hiddenSize) + inputSize;
      } else {
        combinedSize = hiddenSize + hiddenSize;
      }

      W_f_tiers.add(initWeights(combinedSize, hiddenSize));
      W_i_tiers.add(initWeights(combinedSize, hiddenSize));
      W_c_tiers.add(initWeights(combinedSize, hiddenSize));
      W_o_tiers.add(initWeights(combinedSize, hiddenSize));
      b_f_tiers.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      b_i_tiers.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      b_c_tiers.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      b_o_tiers.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
    }

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    int totalSteps = sequence.length;

    List<Tensor<Vector>> h_states = [];
    List<Tensor<Vector>> c_states = [];
    for (int i = 0; i < numTiers; i++) {
      h_states.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      c_states.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
    }

    for (int globalStep = 0; globalStep < totalSteps; globalStep++) {
      Tensor<Vector> x_t = Tensor<Vector>(sequence[globalStep]);

      Tensor<Vector> contextFromHigherTiers;
      if (numTiers > 1) {
        contextFromHigherTiers = concatenateAll(c_states.sublist(1));
      } else {
        contextFromHigherTiers = Tensor<Vector>([]);
      }

      Tensor<Vector> temp_combined = concatenate(h_states[0], contextFromHigherTiers);
      Tensor<Vector> combined_input_lower = concatenate(temp_combined, x_t);

      Map<String, Tensor<Vector>> updatedStates = _lstmStep(combined_input_lower, h_states[0], c_states[0], 0);
      h_states[0] = updatedStates['h']!;
      c_states[0] = updatedStates['c']!;

      for (int i = 1; i < numTiers; i++) {
        if ((globalStep + 1) % cumulativeClockCycles[i - 1] == 0) {
          Tensor<Vector> combined_input_higher = concatenate(h_states[i], h_states[i - 1]);

          Map<String, Tensor<Vector>> updatedHigherStates = _lstmStep(combined_input_higher, h_states[i], c_states[i], i);
          h_states[i] = updatedHigherStates['h']!;
          c_states[i] = updatedHigherStates['c']!;
        }
      }
    }

    return h_states[0];
  }

  Map<String, Tensor<Vector>> _lstmStep(
      Tensor<Vector> combined_input,
      Tensor<Vector> h_prev,
      Tensor<Vector> c_prev,
      int tierIndex
      ) {
    Tensor<Vector> f_t = sigmoid(addVector(matVecMul(W_f_tiers[tierIndex], combined_input), b_f_tiers[tierIndex]));
    Tensor<Vector> i_t = sigmoid(addVector(matVecMul(W_i_tiers[tierIndex], combined_input), b_i_tiers[tierIndex]));
    Tensor<Vector> c_tilde_t = vectorTanh(addVector(matVecMul(W_c_tiers[tierIndex], combined_input), b_c_tiers[tierIndex]));
    Tensor<Vector> c_next = addVector(elementWiseMultiply(f_t, c_prev), elementWiseMultiply(i_t, c_tilde_t));
    Tensor<Vector> o_t = sigmoid(addVector(matVecMul(W_o_tiers[tierIndex], combined_input), b_o_tiers[tierIndex]));
    Tensor<Vector> h_next = elementWiseMultiply(o_t, vectorTanh(c_next));

    return {'h': h_next, 'c': c_next};
  }

  Tensor<Vector> concatenateAll(List<Tensor<Vector>> tensors) {
    if (tensors.isEmpty) return Tensor<Vector>([]);
    if (tensors.length == 1) return tensors[0];

    Tensor<Vector> result = tensors[0];
    for(int i = 1; i < tensors.length; i++){
      result = concatenate(result, tensors[i]);
    }
    return result;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weights = {};

    List<Matrix> serializeMatrixList(List<Tensor<Matrix>> tensorList) {
      List<Matrix> values = [];
      for (Tensor<Matrix> t in tensorList) {
        values.add(t.value);
      }
      return values;
    }

    List<Vector> serializeVectorList(List<Tensor<Vector>> tensorList) {
      List<Vector> values = [];
      for (Tensor<Vector> t in tensorList) {
        values.add(t.value);
      }
      return values;
    }

    weights['W_f_tiers'] = serializeMatrixList(W_f_tiers);
    weights['W_i_tiers'] = serializeMatrixList(W_i_tiers);
    weights['W_c_tiers'] = serializeMatrixList(W_c_tiers);
    weights['W_o_tiers'] = serializeMatrixList(W_o_tiers);

    weights['b_f_tiers'] = serializeVectorList(b_f_tiers);
    weights['b_i_tiers'] = serializeVectorList(b_i_tiers);
    weights['b_c_tiers'] = serializeVectorList(b_c_tiers);
    weights['b_o_tiers'] = serializeVectorList(b_o_tiers);

    return weights;
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {

    void _copyMatrixList(List<Tensor<Matrix>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i++) {
        List<dynamic> newMatrixDynamic = newDataList[i] as List<dynamic>;
        Matrix newMatrix = newMatrixDynamic.map((dynamic row) {
          return (row as List<dynamic>).map((dynamic val) => val as double).toList();
        }).toList();

        Tensor<Matrix> tensor = tensorList[i];
        for (int r = 0; r < tensor.value.length; r++) {
          for (int c = 0; c < tensor.value[r].length; c++) {
            tensor.value[r][c] = newMatrix[r][c];
          }
        }
      }
    }

    void _copyVectorList(List<Tensor<Vector>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i++) {
        List<dynamic> newVector = newDataList[i] as List<dynamic>;
        Tensor<Vector> tensor = tensorList[i];
        for (int j = 0; j < tensor.value.length; j++) {
          tensor.value[j] = newVector[j] as double;
        }
      }
    }

    _copyMatrixList(W_f_tiers, weightsMap['W_f_tiers'] as List<dynamic>);
    _copyMatrixList(W_i_tiers, weightsMap['W_i_tiers'] as List<dynamic>);
    _copyMatrixList(W_c_tiers, weightsMap['W_c_tiers'] as List<dynamic>);
    _copyMatrixList(W_o_tiers, weightsMap['W_o_tiers'] as List<dynamic>);

    _copyVectorList(b_f_tiers, weightsMap['b_f_tiers'] as List<dynamic>);
    _copyVectorList(b_i_tiers, weightsMap['b_i_tiers'] as List<dynamic>);
    _copyVectorList(b_c_tiers, weightsMap['b_c_tiers'] as List<dynamic>);
    _copyVectorList(b_o_tiers, weightsMap['b_o_tiers'] as List<dynamic>);
  }
}


void main() {
  // --- Inline Helper Function for Complex Data ---

  void prepareComplexRnnData({
    required List<Tensor<Matrix>> inputs,
    required List<Tensor<Vector>> targets,
    required int numSamples,
    required int sequenceLength,
    required double startOffset,
  }) {
    Random noiseGenerator = Random();
    for (int i = 0; i < numSamples; i++) {
      Matrix sequence = <Vector>[];
      double start = startOffset + i * 0.5; // More spacing between samples
      for (int j = 0; j < sequenceLength; j++) {
        double timeStep = start + j * 0.1;
        // 1. A slow, strong yearly trend (period of ~365 days)
        double yearlyTrend = sin(timeStep * (2 * pi / 36.5));
        // 2. A medium, clear monthly trend (period of ~30 days)
        double monthlyTrend = 0.5 * cos(timeStep * (2 * pi / 3.0));
        // 3. A noisy, fast weekly pattern (period of 7 days)
        double weeklyNoise = 0.2 * sin(timeStep * (2 * pi / 0.7)) + (noiseGenerator.nextDouble() - 0.5) * 0.1;

        double finalValue = yearlyTrend + monthlyTrend + weeklyNoise;
        sequence.add(<double>[finalValue]);
      }
      inputs.add(Tensor<Matrix>(sequence));

      // Target is the next value in the sequence
      double finalTimeStep = start + sequenceLength * 0.1;
      double nextYearly = sin(finalTimeStep * (2 * pi / 36.5));
      double nextMonthly = 0.5 * cos(finalTimeStep * (2 * pi / 3.0));
      double nextWeekly = 0.2 * sin(finalTimeStep * (2 * pi / 0.7));
      targets.add(Tensor<Vector>(<double>[nextYearly + nextMonthly + nextWeekly]));
    }
  }

  double calculateMSE(SNetwork model, List<Tensor<Matrix>> testX, List<Tensor<Vector>> testY) {
    double totalLoss = 0.0;
    for (int i = 0; i < testX.length; i++) {
      totalLoss += mse(model.predict(testX[i]) as Tensor<Vector>, testY[i]).value;
    }
    return totalLoss / testX.length;
  }

  // Trial runner now accepts a total time limit for the entire training run.
  double runSingleTrial({
    required SNetwork model,
    required List<Tensor<Matrix>> trainX,
    required List<Tensor<Vector>> trainY,
    required List<Tensor<Matrix>> testX,
    required List<Tensor<Vector>> testY,
    required int maxEpochs,
    required double learningRate,
    required Duration? timeLimit,
  }) {
    model.call(trainX[0]);
    SGD optimizer = SGD(model.parameters, learningRate: learningRate);
    Stopwatch trainingStopwatch = Stopwatch()..start();
    int epochsCompleted = 0;

    for (int epoch = 0; epoch < maxEpochs; epoch++) {
      if (timeLimit != null && trainingStopwatch.elapsed > timeLimit) {
        print('  -> Time limit reached. Stopping after $epochsCompleted epochs.');
        break;
      }
      for (int i = 0; i < trainX.length; i++) {
        optimizer.zeroGrad();
        Tensor<Scalar> loss = mse(model.call(trainX[i]) as Tensor<Vector>, trainY[i]);
        loss.backward();
        optimizer.step();
      }
      epochsCompleted++;
    }
    trainingStopwatch.stop();
    print('  -> Trained for $epochsCompleted epochs in ${trainingStopwatch.elapsedMilliseconds}ms.');
    return calculateMSE(model, testX, testY);
  }

  // --- 1. Experiment Configuration ---
  print('🔬 Setting up TIME-FAIR comparison on a complex signal...');
  final List<List<int>> configurationsToTest = [
    [],        // Baseline: Standard LSTMLayer
    [7],       // 2-Tier: Aims to capture weekly patterns
    [30],      // 2-Tier: Aims to capture monthly patterns
    [7, 4],    // 3-Tier: Aims to capture weekly and monthly patterns (~7 days, ~28 days)
  ];

  // --- 2. Global Hyperparameters and Data ---
  int sequenceLength = 60; // Longer sequence to capture multiple cycles
  int hiddenSize = 8;
  int epochsForBaseline = 30;
  double learningRate = 0.03;
  int numTrainSamples = 400;
  int numTestSamples = 50;

  List<Tensor<Matrix>> trainX = [];
  List<Tensor<Vector>> trainY = [];
  List<Tensor<Matrix>> testX = [];
  List<Tensor<Vector>> testY = [];
  prepareComplexRnnData(inputs: trainX, targets: trainY, numSamples: numTrainSamples, sequenceLength: sequenceLength, startOffset: 0.0);
  prepareComplexRnnData(inputs: testX, targets: testY, numSamples: numTestSamples, sequenceLength: sequenceLength, startOffset: numTrainSamples * 0.5 + 50.0);

  print('📊 Complex Data Prepared. Calibrating time budget...');
  print('---');

  // --- 3. Calibration Step ---
  Duration timeBudget;
  SNetwork baselineModel = SNetwork([LSTMLayer(hiddenSize), DenseLayer(1)]);

  print('⏱️  Calibrating time budget with Standard LSTM for $epochsForBaseline epochs...');
  Stopwatch calibrationStopwatch = Stopwatch()..start();
  runSingleTrial(
    model: baselineModel, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
    maxEpochs: epochsForBaseline, learningRate: learningRate, timeLimit: null,
  );
  calibrationStopwatch.stop();
  timeBudget = calibrationStopwatch.elapsed;
  print('✅ Time budget set to: ${timeBudget.inMilliseconds}ms');
  print('---');

  // --- 4. Main Comparison Loop ---
  Map<String, double> results = {};

  for (List<int> clockCycles in configurationsToTest) {
    Layer coreLayer;
    String modelName;
    if (clockCycles.isEmpty) {
      modelName = 'Standard LSTM';
      coreLayer = LSTMLayer(hiddenSize);
    } else {
      modelName = 'MultiTier LSTM (Cycles: $clockCycles)';
      coreLayer = MultiTierLSTMLayer(hiddenSize, tierClockCycles: clockCycles);
    }
    print('🏋️ Training $modelName with a time budget of ${timeBudget.inMilliseconds}ms...');
    SNetwork model = SNetwork([coreLayer, DenseLayer(1)]);
    int maxEpochs = (epochsForBaseline * 3).toInt();

    double finalTestLoss = runSingleTrial(
      model: model, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
      maxEpochs: maxEpochs, learningRate: learningRate, timeLimit: timeBudget,
    );
    print('✅ Finished training. Final Test MSE: ${finalTestLoss.toStringAsFixed(8)}');
    print('---');
    results[modelName] = finalTestLoss;
  }

  // --- 5. Final Results Summary ---
  print('\n\n--- 🏆 FINAL TIME-FAIR COMPARISON RESULTS (Complex Signal) ---');
  print('Total training budget for each model: ${timeBudget.inMilliseconds}ms');
  print('--------------------------------------------------');
  String bestModel = '';
  double lowestLoss = double.infinity;

  results.forEach((modelName, loss) {
    print('${modelName.padRight(35)} | Final Test MSE: ${loss.toStringAsFixed(8)}');
    if (loss < lowestLoss) {
      lowestLoss = loss;
      bestModel = modelName;
    }
  });

  print('--------------------------------------------------');
  print('🥇 Best performing model (given equal time): $bestModel');
}