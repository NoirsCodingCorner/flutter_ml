import 'dart:io';
import 'dart:math';
import 'package:flutter_ml/layertypes/reluLayer.dart';
import 'package:flutter_ml/optimizers/optimizers.dart';
import 'package:flutter_ml/optimizers/sgd.dart';

import '../activationFunctions/relu.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';
import 'denseLayer.dart';
import 'dropout.dart';
import 'layer.dart';
import 'lstmLayer.dart';

/// A self-contained, chained, multi-scale recurrent layer.
///
/// This layer is designed to process a single, long sequence of high-frequency
/// data (e.g., daily) by intelligently creating and processing lower-frequency
/// summaries of the past.
///
/// It encapsulates three internal LSTM processors and two learnable aggregation
/// mechanisms to achieve a hierarchical flow of information:
///
/// 1.  **Aggregation**: The distant past is compressed into "weekly" and "monthly"
///     summary vectors using learnable dense transformations.
/// 2.  **Processing**: Three internal LSTMs process the monthly, weekly, and daily
///     sequences in short, fixed-length loops.
/// 3.  **Chaining**: The context (final hidden state) from the monthly LSTM is used
///     to initialize the weekly LSTM, and the weekly LSTM's context initializes
///     the daily LSTM, creating a deep, sequential chain for backpropagation.
/// A generalized, chained, multi-scale recurrent layer.
///
/// This layer is a fully configurable version of the ChainedMultiScaleLayer.
/// Instead of fixed "daily," "weekly," and "monthly" tiers, it accepts a
/// list of graining sizes to dynamically build a hierarchy of any depth.
///
/// For example, a `grainingSizes` of `[7, 4]` creates a 3-tier system where:
/// - Tier 2 (highest) summarizes chunks of 4 summaries from Tier 1.
/// - Tier 1 summarizes chunks of 7 raw inputs from Tier 0.
/// - Tier 0 (lowest) processes the most recent raw inputs.
class GeneralizedChainedScaleLayer extends Layer {
  @override
  String name = 'generalized_chained_scale';

  final int hiddenSize;
  /// Defines the aggregation factor at each level of the hierarchy.
  /// Example: [7, 4] means the first aggregation level groups 7 inputs,
  /// and the next level groups 4 of those summaries.
  final List<int> grainingSizes;
  /// The number of recent, high-resolution steps for the final LSTM to process.
  final int finalSequenceLength;
  final int numTiers;

  // --- Parameters ---
  // A single list holds the parameters for each tier.
  late List<Tensor<Matrix>> lstmWf, lstmWi, lstmWc, lstmWo;
  late List<Tensor<Vector>> lstmBf, lstmBi, lstmBc, lstmBo;
  late List<Tensor<Matrix>> aggW;
  late List<Tensor<Vector>> aggB;

  GeneralizedChainedScaleLayer({
    required this.hiddenSize,
    required this.grainingSizes,
    this.finalSequenceLength = 7,
  }) : numTiers = grainingSizes.length + 1;

  @override
  List<Tensor> get parameters {
    List<Tensor> allParams = [];
    for (int i = 0; i < numTiers; i++) {
      allParams.addAll([
        lstmWf[i], lstmBf[i], lstmWi[i], lstmBi[i],
        lstmWc[i], lstmBc[i], lstmWo[i], lstmBo[i],
      ]);
    }
    for (int i = 0; i < grainingSizes.length; i++) {
      allParams.addAll([aggW[i], aggB[i]]);
    }
    return allParams;
  }

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    // Initialize parameter lists
    lstmWf = []; lstmWi = []; lstmWc = []; lstmWo = [];
    lstmBf = []; lstmBi = []; lstmBc = []; lstmBo = [];
    aggW = []; aggB = [];

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

    // --- Initialize Aggregation Weights ---
    int currentAggInputSize = inputSize;
    for (int grainSize in grainingSizes) {
      // Input is `grainSize` vectors of size `currentAggInputSize`
      int fanIn = grainSize * currentAggInputSize;
      // Output is a single summary vector of the same size `currentAggInputSize`
      aggW.add(initWeights(fanIn, currentAggInputSize));
      aggB.add(Tensor<Vector>(List<double>.filled(currentAggInputSize, 0.0)));
    }

    // --- Initialize LSTM Weights for Each Tier ---
    for (int i = 0; i < numTiers; i++) {
      int lstmInputFeatureSize = (i == 0) ? inputSize : inputSize;
      int lstmCombinedSize = hiddenSize + lstmInputFeatureSize;
      lstmWf.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWi.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWc.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWo.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmBf.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBi.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBc.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBo.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
    }

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix fullSequence = (input as Tensor<Matrix>).value;

    // --- 1. DATA PREPARATION AND AGGREGATION (from oldest to newest) ---
    List<List<Tensor<Vector>>> tierInputs = List.generate(numTiers, (_) => []);

    // The final, highest-resolution data for the last LSTM
    int lastTierStart = fullSequence.length - finalSequenceLength;
    Matrix finalDataRaw = fullSequence.sublist(lastTierStart);
    tierInputs[0] = finalDataRaw.map((v) => Tensor<Vector>(v)).toList();

    // Iteratively build the aggregated inputs for higher tiers
    List<Tensor<Vector>> currentLevelData = fullSequence.map((v) => Tensor<Vector>(v)).toList();
    for (int i = 0; i < grainingSizes.length; i++) {
      int grainSize = grainingSizes[i];
      List<Tensor<Vector>> nextLevelSummaries = [];
      for (int j = 0; j <= currentLevelData.length - grainSize; j += grainSize) {
        List<Tensor<Vector>> chunk = currentLevelData.sublist(j, j + grainSize);
        Tensor<Vector> flattened = concatenateAll(chunk);
        Tensor<Vector> summary = vectorTanh(add(matVecMul(aggW[i], flattened), aggB[i]));
        nextLevelSummaries.add(summary);
      }
      // The summaries from this level become the input for the next level up in the hierarchy.
      tierInputs[i + 1] = nextLevelSummaries;
      currentLevelData = nextLevelSummaries;
    }

    // --- 2. HIERARCHICAL LSTM PROCESSING (from highest/oldest tier to lowest/newest) ---
    Tensor<Vector> context = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)); // Initial empty context

    for (int i = numTiers - 1; i >= 0; i--) {
      List<Tensor<Vector>> sequenceForThisTier = tierInputs[i];
      // We only need a fixed number of recent steps for each tier's LSTM
      int lookback = (i == 0) ? finalSequenceLength : 10; // Use a fixed lookback for higher tiers
      List<Tensor<Vector>> finalSequence = sequenceForThisTier.sublist(
          max(0, sequenceForThisTier.length - lookback)
      );

      context = _lstmLoop(
        finalSequence,
        context, // The output of the higher tier becomes the initial state for this one
        i, // Pass the tier index to use the correct weights
      );
    }

    return context; // The final output from the lowest, most detailed tier
  }

  /// A private helper to run a standard LSTM loop for a specific tier.
  Tensor<Vector> _lstmLoop(
      List<Tensor<Vector>> sequence,
      Tensor<Vector> initialState,
      int tierIndex,
      ) {
    Tensor<Vector> h = initialState;
    Tensor<Vector> c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Tensor<Vector> x_t in sequence) {
      Tensor<Vector> combined_input = concatenate(h, x_t);

      Tensor<Vector> f_t = sigmoid(add(matVecMul(lstmWf[tierIndex], combined_input), lstmBf[tierIndex]));
      Tensor<Vector> i_t = sigmoid(add(matVecMul(lstmWi[tierIndex], combined_input), lstmBi[tierIndex]));
      Tensor<Vector> c_tilde_t = vectorTanh(add(matVecMul(lstmWc[tierIndex], combined_input), lstmBc[tierIndex]));

      c = add(elementWiseMultiply(f_t, c), elementWiseMultiply(i_t, c_tilde_t));

      Tensor<Vector> o_t = sigmoid(add(matVecMul(lstmWo[tierIndex], combined_input), lstmBo[tierIndex]));
      h = elementWiseMultiply(o_t, vectorTanh(c));
    }
    return h;
  }

  // Helper, assumed to be available
  Tensor<Vector> concatenateAll(List<Tensor<Vector>> tensors) {
    if (tensors.isEmpty) return Tensor<Vector>([]);
    if (tensors.length == 1) return tensors[0];
    Tensor<Vector> result = tensors[0];
    for (int i = 1; i < tensors.length; i++) {
      result = concatenate(result, tensors[i]);
    }
    return result;
  }
}

// --- A NEW, NECESSARY Reshape Layer ---
// This layer converts a Vector of length 1 into a 1x1 Matrix.
class ReshapeVectorToMatrixLayer extends Layer {
  @override
  String name = 'reshape_vec_to_mat';
  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Vector v = (input as Tensor<Vector>).value;
    // Reshapes a vector like [x] into a matrix like [[x]]
    return reshapeVectorToMatrix(input, 1, 1);
  }
}


// --- NEW Helper Function to Control Dropout ---
/// Iterates through a model's layers and sets the `isTraining` flag
/// on any `DropoutLayer` instances.
void setTrainingMode(SNetwork model, bool isTraining) {
  for (Layer layer in model.layers) {
    if (layer is DropoutLayer) {
      layer.isTraining = isTraining;
    }
  }
}

void main() {
  // --- 1. File Loading and Parsing ---
  print('💾 Loading silver price data...');
  File csvFile = File('lib/datasets/silver.csv');
  if (!csvFile.existsSync()) {
    print('Error: silver_price.csv not found at the specified path.');
    return;
  }
  List<String> lines = csvFile.readAsLinesSync();
  List<double> rawPricesReversed = [];
  for (int i = 1; i < lines.length; i++) {
    List<String> parts = lines[i].replaceAll('"', '').split(',');
    if (parts.length > 1 && double.tryParse(parts[1]) != null) {
      rawPricesReversed.add(double.parse(parts[1]));
    }
  }
  List<double> allPrices = rawPricesReversed.reversed.toList();
  print('Loaded and processed ${allPrices.length} data points.');
  print('---');

  // --- 2. Data Pre-processing using Differencing ---
  print('🛠️  Pre-processing data using differencing...');
  List<double> priceChanges = [];
  for (int i = 1; i < allPrices.length; i++) {
    priceChanges.add(allPrices[i] - allPrices[i - 1]);
  }
  double changesMin = priceChanges.reduce(min);
  double changesMax = priceChanges.reduce(max);
  List<double> changesNormalized = priceChanges.map((c) => (c - changesMin) / (changesMax - changesMin)).toList();

  // --- 3. Splitting and Preparing Final Datasets ---
  int sequenceLength = 120;
  int splitIndex = (changesNormalized.length * 0.8).floor();
  List<Tensor<Matrix>> trainX = [];
  List<Tensor<Vector>> trainY = [];
  for (int i = 0; i < splitIndex - sequenceLength; i++) {
    Matrix sequence = [];
    for(int j = 0; j < sequenceLength; j++) {
      sequence.add([changesNormalized[i + j]]);
    }
    trainX.add(Tensor<Matrix>(sequence));
    trainY.add(Tensor<Vector>([changesNormalized[i + sequenceLength]]));
  }
  print('Prepared ${trainX.length} training samples and ${changesNormalized.length - splitIndex} test samples.');
  print('---');

  // --- 4. Ensemble Training Loop ---
  int numModels = 4;
  List<SNetwork> trainedModels = [];
  int epochs = 8;
  double learningRate = 0.001;

  for (int modelIndex = 0; modelIndex < numModels; modelIndex++) {
    print('🧠 --- Training Model ${modelIndex + 1}/$numModels ---');

    // *** MODIFIED MODEL DEFINITION ***
    SNetwork model = SNetwork([
      GeneralizedChainedScaleLayer(hiddenSize: 64, grainingSizes: [30, 15, 7, 4], finalSequenceLength: 14),
      DropoutLayer(0.25), // Add a Dropout layer with a 25% rate
      DenseLayer(1),
    ]);

    model.predict(trainX[0]);
    SGD optimizer = SGD(model.parameters, learningRate: learningRate);

    // Set model to training mode (activates dropout)
    setTrainingMode(model, true);

    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0;
      for (int i = 0; i < trainX.length; i++) {
        optimizer.zeroGrad();
        Tensor<Vector> pred_norm = model.call(trainX[i]) as Tensor<Vector>;
        Tensor<Scalar> loss = mse(pred_norm, trainY[i]);
        loss.backward();
        optimizer.step();
        totalLoss += loss.value;
      }
      print('  -> Model ${modelIndex+1}, Epoch ${(epoch + 1)}/${epochs} | Avg Loss: ${totalLoss / trainX.length}');
    }
    trainedModels.add(model);
    print('✅ Model ${modelIndex + 1} training finished.');
  }
  print('---');

  // --- 5. Evaluation on Test Set ---
  print('📈 Evaluating model ensemble on the test set...');

  // *** SET ALL MODELS TO EVALUATION MODE ***
  // This deactivates dropout for consistent predictions.
  for (SNetwork model in trainedModels) {
    setTrainingMode(model, false);
  }

  double totalAbsoluteError = 0;
  int correctDirectionCount = 0;
  int testSamplesCount = changesNormalized.length - splitIndex - sequenceLength;

  for (int i = 0; i < testSamplesCount; i++) {
    int currentIndex = splitIndex + i;
    double lastKnownPrice = allPrices[currentIndex + sequenceLength - 1];
    double actualPrice = allPrices[currentIndex + sequenceLength];
    Matrix inputSequence = [];
    for (int j = 0; j < sequenceLength; j++) {
      inputSequence.add([changesNormalized[currentIndex + j]]);
    }
    Tensor<Matrix> inputTensor = Tensor<Matrix>(inputSequence);

    double ensemblePredictedChange = 0;
    for(SNetwork model in trainedModels){
      Tensor<Vector> pred_norm = model.predict(inputTensor) as Tensor<Vector>;
      ensemblePredictedChange += pred_norm.value[0];
    }
    double avg_pred_norm = ensemblePredictedChange / numModels;
    double predictedChange = avg_pred_norm * (changesMax - changesMin) + changesMin;
    double finalPrediction = lastKnownPrice + predictedChange;

    totalAbsoluteError += (finalPrediction - actualPrice).abs();

    double actualChange = actualPrice - lastKnownPrice;
    if ((predictedChange >= 0 && actualChange >= 0) || (predictedChange < 0 && actualChange < 0)) {
      correctDirectionCount++;
    }
  }

  double meanAbsoluteError = totalAbsoluteError / testSamplesCount;
  double directionalAccuracy = (correctDirectionCount / testSamplesCount) * 100;

  print('\n--- FINAL PERFORMANCE ON TEST SET ---');
  print('Mean Absolute Error (MAE): \$${meanAbsoluteError.toStringAsFixed(4)}');
  print('Directional Accuracy: ${directionalAccuracy.toStringAsFixed(2)}%');
  print('-------------------------------------\n');

  // --- 6. Ensemble Forecasting for the Next Month ---
  print('🔮 Generating ensemble forecast for the next 30 days...');
  int forecastDays = 30;
  List<List<double>> allForecasts = [];

  // Models are already in evaluation mode, so no change is needed here.
  for (SNetwork model in trainedModels) {
    List<double> forecast = [];
    List<double> currentSequence = changesNormalized.sublist(changesNormalized.length - sequenceLength);
    double lastPrice = allPrices.last;

    for (int day = 0; day < forecastDays; day++) {
      Matrix inputMatrix = [];
      for(double change in currentSequence) {
        inputMatrix.add([change]);
      }
      Tensor<Vector> nextChangeNorm = model.predict(Tensor<Matrix>(inputMatrix)) as Tensor<Vector>;
      double nextChange = nextChangeNorm.value[0] * (changesMax - changesMin) + changesMin;
      double nextPrice = lastPrice + nextChange;
      forecast.add(nextPrice);

      currentSequence.removeAt(0);
      currentSequence.add(nextChangeNorm.value[0]);
      lastPrice = nextPrice;
    }
    allForecasts.add(forecast);
  }

  // --- 7. Analyzing and Displaying the Ensemble Forecast ---
  print('\n--- 🗓️ 30-Day Silver Price Ensemble Forecast ---');
  // ... (This section remains the same)
  for (int day = 0; day < forecastDays; day++) {
    List<double> predictionsForDay = allForecasts.map((forecast) => forecast[day]).toList();
    double sum = predictionsForDay.reduce((a, b) => a + b);
    double average = sum / numModels;
    double minPred = predictionsForDay.reduce(min);
    double maxPred = predictionsForDay.reduce(max);
    String dayStr = (day + 1).toString().padLeft(2);
    String avgStr = average.toStringAsFixed(4).padLeft(9);
    String minStr = minPred.toStringAsFixed(4);
    String maxStr = maxPred.toStringAsFixed(4);
    print('  Day $dayStr: Avg Price: \$$avgStr (Range: \$${minStr} - \$${maxStr})');
  }
}

