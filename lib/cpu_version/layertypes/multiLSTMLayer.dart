import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class MultiTierLSTMLayer extends Layer<Matrix, Vector> {
  @override
  String name = 'multitier_lstm';

  int hiddenSize;
  List<int> tierClockCycles;
  late int numTiers;

  late List<Tensor<Matrix>> W_f_tiers;
  late List<Tensor<Matrix>> W_i_tiers;
  late List<Tensor<Matrix>> W_c_tiers;
  late List<Tensor<Matrix>> W_o_tiers;

  late List<Tensor<Vector>> b_f_tiers;
  late List<Tensor<Vector>> b_i_tiers;
  late List<Tensor<Vector>> b_c_tiers;
  late List<Tensor<Vector>> b_o_tiers;

  late List<int> cumulativeClockCycles;

  MultiTierLSTMLayer(this.hiddenSize, {required this.tierClockCycles}) {
    numTiers = tierClockCycles.length + 1;
    cumulativeClockCycles = [];
    int product = 1;
    for (int i = 0; i < tierClockCycles.length; i = i + 1) {
      product = product * tierClockCycles[i];
      cumulativeClockCycles.add(product);
    }
  }

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> allParams = [];
    for (int i = 0; i < numTiers; i = i + 1) {
      allParams.add(W_f_tiers[i]);
      allParams.add(b_f_tiers[i]);
      allParams.add(W_i_tiers[i]);
      allParams.add(b_i_tiers[i]);
      allParams.add(W_c_tiers[i]);
      allParams.add(b_c_tiers[i]);
      allParams.add(W_o_tiers[i]);
      allParams.add(b_o_tiers[i]);
    }
    return allParams;
  }

  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int inputSize = 0;
    if (inputMatrix.isNotEmpty) {
      inputSize = inputMatrix[0].length;
    }
    Random random = Random();

    W_f_tiers = []; W_i_tiers = []; W_c_tiers = []; W_o_tiers = [];
    b_f_tiers = []; b_i_tiers = []; b_c_tiers = []; b_o_tiers = [];

    Tensor<Matrix> initWeights(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      Matrix values = [];
      for (int i = 0; i < fanOut; i = i + 1) {
        Vector row = [];
        for (int j = 0; j < fanIn; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    Tensor<Vector> initBias() {
      Vector values = [];
      for (int i = 0; i < hiddenSize; i = i + 1) {
        values.add(0.0);
      }
      return Tensor<Vector>(values);
    }

    for (int i = 0; i < numTiers; i = i + 1) {
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

      b_f_tiers.add(initBias());
      b_i_tiers.add(initBias());
      b_c_tiers.add(initBias());
      b_o_tiers.add(initBias());
    }

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    Matrix sequence = input.value;
    int totalSteps = sequence.length;

    List<Tensor<Vector>> hStates = [];
    List<Tensor<Vector>> cStates = [];

    for (int i = 0; i < numTiers; i = i + 1) {
      Vector zeroVectorH = [];
      Vector zeroVectorC = [];
      for (int j = 0; j < hiddenSize; j = j + 1) {
        zeroVectorH.add(0.0);
        zeroVectorC.add(0.0);
      }
      hStates.add(Tensor<Vector>(zeroVectorH));
      cStates.add(Tensor<Vector>(zeroVectorC));
    }

    for (int globalStep = 0; globalStep < totalSteps; globalStep = globalStep + 1) {
      Tensor<Vector> xT = Tensor<Vector>(sequence[globalStep]);

      Tensor<Vector> contextFromHigherTiers;
      if (numTiers > 1) {
        List<Tensor<Vector>> sublist = [];
        for (int j = 1; j < cStates.length; j = j + 1) {
          sublist.add(cStates[j]);
        }
        contextFromHigherTiers = concatenateAll(sublist);
      } else {
        contextFromHigherTiers = Tensor<Vector>([]);
      }

      Tensor<Vector> tempCombined = concatenate(hStates[0], contextFromHigherTiers);
      Tensor<Vector> combinedInputLower = concatenate(tempCombined, xT);

      Map<String, Tensor<Vector>> updatedStates = _lstmStep(combinedInputLower, hStates[0], cStates[0], 0);
      hStates[0] = updatedStates['h']!;
      cStates[0] = updatedStates['c']!;

      for (int i = 1; i < numTiers; i = i + 1) {
        if ((globalStep + 1) % cumulativeClockCycles[i - 1] == 0) {
          Tensor<Vector> combinedInputHigher = concatenate(hStates[i], hStates[i - 1]);

          Map<String, Tensor<Vector>> updatedHigherStates = _lstmStep(combinedInputHigher, hStates[i], cStates[i], i);
          hStates[i] = updatedHigherStates['h']!;
          cStates[i] = updatedHigherStates['c']!;
        }
      }
    }

    return hStates[0];
  }

  Map<String, Tensor<Vector>> _lstmStep(
      Tensor<Vector> combinedInput,
      Tensor<Vector> hPrev,
      Tensor<Vector> cPrev,
      int tierIndex
      ) {
    Tensor<Vector> fT = sigmoid(addVector(matVecMul(W_f_tiers[tierIndex], combinedInput), b_f_tiers[tierIndex]));
    Tensor<Vector> iT = sigmoid(addVector(matVecMul(W_i_tiers[tierIndex], combinedInput), b_i_tiers[tierIndex]));
    Tensor<Vector> cTildeT = vectorTanh(addVector(matVecMul(W_c_tiers[tierIndex], combinedInput), b_c_tiers[tierIndex]));
    Tensor<Vector> cNext = addVector(elementWiseMultiply(fT, cPrev), elementWiseMultiply(iT, cTildeT));
    Tensor<Vector> oT = sigmoid(addVector(matVecMul(W_o_tiers[tierIndex], combinedInput), b_o_tiers[tierIndex]));
    Tensor<Vector> hNext = elementWiseMultiply(oT, vectorTanh(cNext));

    return {'h': hNext, 'c': cNext};
  }

  Tensor<Vector> concatenateAll(List<Tensor<Vector>> tensors) {
    if (tensors.isEmpty) {
      return Tensor<Vector>([]);
    }
    if (tensors.length == 1) {
      return tensors[0];
    }

    Tensor<Vector> result = tensors[0];
    for (int i = 1; i < tensors.length; i = i + 1) {
      result = concatenate(result, tensors[i]);
    }
    return result;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weights = {};

    List<Matrix> serializeMatrixList(List<Tensor<Matrix>> tensorList) {
      List<Matrix> values = [];
      for (int i = 0; i < tensorList.length; i = i + 1) {
        values.add(tensorList[i].value);
      }
      return values;
    }

    List<Vector> serializeVectorList(List<Tensor<Vector>> tensorList) {
      List<Vector> values = [];
      for (int i = 0; i < tensorList.length; i = i + 1) {
        values.add(tensorList[i].value);
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

    void copyMatrixList(List<Tensor<Matrix>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i = i + 1) {
        List<dynamic> newMatrixDynamic = newDataList[i] as List<dynamic>;
        Tensor<Matrix> tensor = tensorList[i];

        int idx = 0;
        for (int r = 0; r < newMatrixDynamic.length; r = r + 1) {
          List<dynamic> rowDynamic = newMatrixDynamic[r] as List<dynamic>;
          for (int c = 0; c < rowDynamic.length; c = c + 1) {
            tensor.data[idx] = rowDynamic[c] as double;
            idx = idx + 1;
          }
        }
      }
    }

    void copyVectorList(List<Tensor<Vector>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i = i + 1) {
        List<dynamic> newVectorDynamic = newDataList[i] as List<dynamic>;
        Tensor<Vector> tensor = tensorList[i];

        for (int j = 0; j < newVectorDynamic.length; j = j + 1) {
          tensor.data[j] = newVectorDynamic[j] as double;
        }
      }
    }

    copyMatrixList(W_f_tiers, weightsMap['W_f_tiers'] as List<dynamic>);
    copyMatrixList(W_i_tiers, weightsMap['W_i_tiers'] as List<dynamic>);
    copyMatrixList(W_c_tiers, weightsMap['W_c_tiers'] as List<dynamic>);
    copyMatrixList(W_o_tiers, weightsMap['W_o_tiers'] as List<dynamic>);

    copyVectorList(b_f_tiers, weightsMap['b_f_tiers'] as List<dynamic>);
    copyVectorList(b_i_tiers, weightsMap['b_i_tiers'] as List<dynamic>);
    copyVectorList(b_c_tiers, weightsMap['b_c_tiers'] as List<dynamic>);
    copyVectorList(b_o_tiers, weightsMap['b_o_tiers'] as List<dynamic>);
  }
}

// ─────────────────────────────────────────────────────── //
// MAIN TESTING FUNCTION
// ─────────────────────────────────────────────────────── //

/*void main() {
  void prepareComplexRnnData({
    required List<Tensor<Matrix>> inputs,
    required List<Tensor<Vector>> targets,
    required int numSamples,
    required int sequenceLength,
    required double startOffset,
  }) {
    Random noiseGenerator = Random();
    for (int i = 0; i < numSamples; i = i + 1) {
      Matrix sequence = [];
      double start = startOffset + i * 0.5;

      for (int j = 0; j < sequenceLength; j = j + 1) {
        double timeStep = start + j * 0.1;
        double yearlyTrend = sin(timeStep * (2.0 * pi / 36.5));
        double monthlyTrend = 0.5 * cos(timeStep * (2.0 * pi / 3.0));
        double weeklyNoise = 0.2 * sin(timeStep * (2.0 * pi / 0.7)) + (noiseGenerator.nextDouble() - 0.5) * 0.1;

        double finalValue = yearlyTrend + monthlyTrend + weeklyNoise;

        Vector singleFeature = [];
        singleFeature.add(finalValue);
        sequence.add(singleFeature);
      }
      inputs.add(Tensor<Matrix>(sequence));

      double finalTimeStep = start + sequenceLength * 0.1;
      double nextYearly = sin(finalTimeStep * (2.0 * pi / 36.5));
      double nextMonthly = 0.5 * cos(finalTimeStep * (2.0 * pi / 3.0));
      double nextWeekly = 0.2 * sin(finalTimeStep * (2.0 * pi / 0.7));

      Vector targetFeature = [];
      targetFeature.add(nextYearly + nextMonthly + nextWeekly);
      targets.add(Tensor<Vector>(targetFeature));
    }
  }

  double calculateMSE(SNetwork model, List<Tensor<Matrix>> testX, List<Tensor<Vector>> testY) {
    double totalLoss = 0.0;
    for (int i = 0; i < testX.length; i = i + 1) {
      Tensor<Vector> prediction = model.predict(testX[i]) as Tensor<Vector>;
      Tensor<Scalar> lossVal = mse(prediction, testY[i]);
      totalLoss = totalLoss + lossVal.value;
    }
    return totalLoss / testX.length;
  }

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
    Stopwatch trainingStopwatch = Stopwatch();
    trainingStopwatch.start();

    int epochsCompleted = 0;

    for (int epoch = 0; epoch < maxEpochs; epoch = epoch + 1) {
      if (timeLimit != null && trainingStopwatch.elapsed > timeLimit) {
        print('  -> Time limit reached. Stopping after $epochsCompleted epochs.');
        break;
      }
      for (int i = 0; i < trainX.length; i = i + 1) {
        optimizer.zeroGrad();
        Tensor<Vector> prediction = model.call(trainX[i]) as Tensor<Vector>;
        Tensor<Scalar> loss = mse(prediction, trainY[i]);
        loss.backward();
        optimizer.step();
      }
      epochsCompleted = epochsCompleted + 1;
    }
    trainingStopwatch.stop();
    print('  -> Trained for $epochsCompleted epochs in ${trainingStopwatch.elapsedMilliseconds}ms.');
    return calculateMSE(model, testX, testY);
  }

  print('🔬 Setting up TIME-FAIR comparison on a complex signal...');

  List<List<int>> configurationsToTest = [];
  configurationsToTest.add([]);        // Baseline: Standard LSTMLayer
  configurationsToTest.add([7]);       // 2-Tier: Aims to capture weekly patterns
  configurationsToTest.add([30]);      // 2-Tier: Aims to capture monthly patterns
  configurationsToTest.add([7, 4]);    // 3-Tier: Aims to capture weekly and monthly patterns

  int sequenceLength = 10;
  int hiddenSize = 8;
  int epochsForBaseline = 10;
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

  Duration timeBudget = Duration.zero;
  List<Layer<dynamic, dynamic>> baselineLayers = [];
  baselineLayers.add(LSTMLayer(hiddenSize));
  baselineLayers.add(DenseLayer(1));
  SNetwork baselineModel = SNetwork(baselineLayers);

  print('⏱️  Calibrating time budget with Standard LSTM for $epochsForBaseline epochs...');
  Stopwatch calibrationStopwatch = Stopwatch();
  calibrationStopwatch.start();

  runSingleTrial(
    model: baselineModel, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
    maxEpochs: epochsForBaseline, learningRate: learningRate, timeLimit: null,
  );

  calibrationStopwatch.stop();
  timeBudget = calibrationStopwatch.elapsed;
  print('✅ Time budget set to: ${timeBudget.inMilliseconds}ms');
  print('---');

  Map<String, double> results = {};

  for (int c = 0; c < configurationsToTest.length; c = c + 1) {
    List<int> clockCycles = configurationsToTest[c];
    Layer<dynamic, dynamic> coreLayer;
    String modelName = '';

    if (clockCycles.isEmpty) {
      modelName = 'Standard LSTM';
      coreLayer = LSTMLayer(hiddenSize);
    } else {
      modelName = 'MultiTier LSTM (Cycles: $clockCycles)';
      coreLayer = MultiTierLSTMLayer(hiddenSize, tierClockCycles: clockCycles);
    }

    print('🏋️ Training $modelName with a time budget of ${timeBudget.inMilliseconds}ms...');

    List<Layer<dynamic, dynamic>> currentLayers = [];
    currentLayers.add(coreLayer);
    currentLayers.add(DenseLayer(1));
    SNetwork model = SNetwork(currentLayers);

    int maxEpochs = epochsForBaseline * 3;

    double finalTestLoss = runSingleTrial(
      model: model, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
      maxEpochs: maxEpochs, learningRate: learningRate, timeLimit: timeBudget,
    );

    print('✅ Finished training. Final Test MSE: ${finalTestLoss.toStringAsFixed(8)}');
    print('---');
    results[modelName] = finalTestLoss;
  }

  print('\n\n--- 🏆 FINAL TIME-FAIR COMPARISON RESULTS (Complex Signal) ---');
  print('Total training budget for each model: ${timeBudget.inMilliseconds}ms');
  print('--------------------------------------------------');

  String bestModel = '';
  double lowestLoss = double.infinity;

  List<String> keys = results.keys.toList();
  for (int k = 0; k < keys.length; k = k + 1) {
    String modelName = keys[k];
    double loss = results[modelName]!;
    print('${modelName.padRight(35)} | Final Test MSE: ${loss.toStringAsFixed(8)}');
    if (loss < lowestLoss) {
      lowestLoss = loss;
      bestModel = modelName;
    }
  }

  print('--------------------------------------------------');
  print('🥇 Best performing model (given equal time): $bestModel');
}*/