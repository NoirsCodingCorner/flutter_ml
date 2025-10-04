import 'dart:io';
import 'dart:math';

import 'package:flutter_ml/autogradEngine/tensor.dart';
import 'package:flutter_ml/optimizers/sgd.dart';

import '../layertypes/denseLayer.dart';
import '../layertypes/dropout.dart';
import '../layertypes/trendmodelLayer.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';


import 'dart:io';
import 'dart:math';

// Assume you already imported your ML layers and Tensor types like SNetwork, Tensor, etc.

class SilverForecaster {
  // --- Configuration ---
  final String inputCsvPath;
  final String outputCsvPath;
  final int sequenceLength;
  final int hiddenSize;
  final List<int> grainingSizes;
  final double dropoutRate;
  final int numModels;
  final int epochs;
  final double learningRate;

  // --- Internal State ---
  late List<double> _allPrices;
  late List<String> _allDates; // NEW: to store dates
  late List<double> _changesNormalized;
  late double _changesMin;
  late double _changesMax;
  late List<SNetwork> _trainedModels;

  SilverForecaster({
    required this.inputCsvPath,
    required this.outputCsvPath,
    this.sequenceLength = 60,
    this.hiddenSize = 26,
    this.grainingSizes = const [7, 4],
    this.dropoutRate = 0.25,
    this.numModels = 4,
    this.epochs = 8,
    this.learningRate = 0.001,
  });

  /// The main public method to run the entire forecasting process.
  void run() {
    if (!_loadAndParseData()) return;
    _preprocessData();

    int splitIndex = (_changesNormalized.length * 0.8).floor();
    var (trainX, trainY) = _prepareTrainingData(splitIndex);

    _trainEnsemble(trainX, trainY);
    _evaluate(splitIndex);
    _forecast();
  }

  bool _loadAndParseData() {
    print('💾 Loading silver price data from $inputCsvPath...');
    File csvFile = File(inputCsvPath);
    if (!csvFile.existsSync()) {
      print('Error: CSV file not found at the specified path.');
      return false;
    }

    List<String> lines = csvFile.readAsLinesSync();
    List<double> rawPricesReversed = [];
    List<String> rawDatesReversed = [];

    for (int i = 1; i < lines.length; i++) {
      List<String> parts = lines[i].replaceAll('"', '').split(',');
      if (parts.length > 1 && double.tryParse(parts[1]) != null) {
        rawDatesReversed.add(parts[0]); // first column = date
        rawPricesReversed.add(double.parse(parts[1]));
      }
    }

    _allPrices = rawPricesReversed.reversed.toList();
    _allDates = rawDatesReversed.reversed.toList();

    print('Loaded and processed ${_allPrices.length} data points.');
    print('---');
    return true;
  }

  void _preprocessData() {
    print('🛠️  Pre-processing data using differencing...');
    List<double> priceChanges = [];
    for (int i = 1; i < _allPrices.length; i++) {
      priceChanges.add(_allPrices[i] - _allPrices[i - 1]);
    }
    _changesMin = priceChanges.reduce(min);
    _changesMax = priceChanges.reduce(max);
    _changesNormalized = priceChanges
        .map((c) => (c - _changesMin) / (_changesMax - _changesMin))
        .toList();
  }

  (List<Tensor<Matrix>>, List<Tensor<Vector>>) _prepareTrainingData(int splitIndex) {
    List<Tensor<Matrix>> trainX = [];
    List<Tensor<Vector>> trainY = [];
    for (int i = 0; i < splitIndex - sequenceLength; i++) {
      Matrix sequence = [];
      for (int j = 0; j < sequenceLength; j++) {
        sequence.add([_changesNormalized[i + j]]);
      }
      trainX.add(Tensor<Matrix>(sequence));
      trainY.add(Tensor<Vector>([_changesNormalized[i + sequenceLength]]));
    }
    print('Prepared ${trainX.length} training samples and '
        '${_changesNormalized.length - splitIndex} test samples.');
    print('---');
    return (trainX, trainY);
  }

  void _trainEnsemble(List<Tensor<Matrix>> trainX, List<Tensor<Vector>> trainY) {
    _trainedModels = [];
    for (int modelIndex = 0; modelIndex < numModels; modelIndex++) {
      print('🧠 --- Training Model ${modelIndex + 1}/$numModels ---');
      SNetwork model = SNetwork([
        GeneralizedChainedScaleLayer(
            hiddenSize: hiddenSize,
            grainingSizes: grainingSizes,
            finalSequenceLength: 14),
        DropoutLayer(dropoutRate),
        DenseLayer(1),
      ]);
      model.predict(trainX[0]);
      SGD optimizer = SGD(model.parameters, learningRate: learningRate);
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
        print(
            '  -> Model ${modelIndex + 1}, Epoch ${(epoch + 1)}/$epochs | Avg Loss: ${totalLoss / trainX.length}');
      }
      _trainedModels.add(model);
      print('✅ Model ${modelIndex + 1} training finished.');
    }
    print('---');
  }

  void _evaluate(int splitIndex, {int horizon = 20}) {
    print('📈 Evaluating model ensemble with multi-day horizons...');
    for (SNetwork model in _trainedModels) {
      setTrainingMode(model, false);
    }

    // CSV with Actuals + rolling forecasts
    StringBuffer csvContent = StringBuffer("Date,ActualPrice,PredictedH1,PredictedH2,...,PredictedH$horizon\n");

    double totalAbsoluteError = 0;
    int testSamplesCount = _changesNormalized.length - splitIndex - sequenceLength - horizon;

    for (int i = 0; i < testSamplesCount; i++) {
      int currentIndex = splitIndex + i;

      // last known price + sequence
      double lastPrice = _allPrices[currentIndex + sequenceLength - 1];
      List<double> currentSeq =
      _changesNormalized.sublist(currentIndex, currentIndex + sequenceLength);

      // collect multi-day forecast
      List<double> forecastPath = [];
      for (int h = 0; h < horizon; h++) {
        // build input tensor
        Matrix inputMatrix = currentSeq.map((c) => [c]).toList();
        Tensor<Matrix> inputTensor = Tensor<Matrix>(inputMatrix);

        // ensemble prediction
        double ensemblePredChange = 0;
        for (SNetwork model in _trainedModels) {
          Tensor<Vector> predNorm = model.predict(inputTensor) as Tensor<Vector>;
          ensemblePredChange += predNorm.value[0];
        }
        double avgNorm = ensemblePredChange / numModels;
        double nextChange = avgNorm * (_changesMax - _changesMin) + _changesMin;

        // update price
        double nextPrice = lastPrice + nextChange;
        forecastPath.add(nextPrice);

        // roll sequence
        currentSeq.removeAt(0);
        currentSeq.add(avgNorm);
        lastPrice = nextPrice;
      }

      // Actual price at forecast horizon
      String date = _allDates[currentIndex + sequenceLength]; // base date
      double actualPrice = _allPrices[currentIndex + sequenceLength];

      // Write row: date, actual, then horizon steps
      csvContent.write("$date,$actualPrice");
      for (int h = 0; h < horizon; h++) {
        csvContent.write(",${forecastPath[h]}");
        if (h < horizon - 1) csvContent.write("");
      }
      csvContent.writeln();

      // error metric example: 1-day MAE
      totalAbsoluteError += (forecastPath[0] - actualPrice).abs();
    }

    String evalPath = outputCsvPath.replaceAll(".csv", "_eval_multihorizon.csv");
    File(evalPath).writeAsStringSync(csvContent.toString());

    double meanAbsoluteError = totalAbsoluteError / testSamplesCount;
    print('📄 Multi-horizon eval exported to: $evalPath');
    print('MAE (1-day ahead only): $meanAbsoluteError');

  }

  void _forecast() {
    print('🔮 Generating ensemble forecast for the next 30 days...');
    int forecastDays = 30;
    List<List<double>> allForecasts = [];

    for (SNetwork model in _trainedModels) {
      List<double> forecast = [];
      List<double> currentSequence =
      _changesNormalized.sublist(_changesNormalized.length - sequenceLength);
      double lastPrice = _allPrices.last;

      for (int day = 0; day < forecastDays; day++) {
        Matrix inputMatrix = [];
        for (double change in currentSequence) {
          inputMatrix.add([change]);
        }
        Tensor<Vector> nextChangeNorm =
        model.predict(Tensor<Matrix>(inputMatrix)) as Tensor<Vector>;
        double nextChange =
            nextChangeNorm.value[0] * (_changesMax - _changesMin) + _changesMin;
        double nextPrice = lastPrice + nextChange;
        forecast.add(nextPrice);

        // roll the sequence forward
        currentSequence.removeAt(0);
        currentSequence.add(nextChangeNorm.value[0]);
        lastPrice = nextPrice;
      }
      allForecasts.add(forecast);
    }

    // Average ensemble forecast
    List<double> avgForecast = [];
    for (int day = 0; day < forecastDays; day++) {
      double sum = 0;
      for (int m = 0; m < numModels; m++) {
        sum += allForecasts[m][day];
      }
      avgForecast.add(sum / numModels);
    }

    // Export to CSV
    String forecastCsv = "Day,ForecastedPrice\n";
    for (int day = 0; day < forecastDays; day++) {
      forecastCsv += "${day + 1},${avgForecast[day]}\n";
    }

    File(outputCsvPath.replaceAll(".csv", "_forecast.csv"))
        .writeAsStringSync(forecastCsv);

    print('📄 30-day forecast exported to ${outputCsvPath.replaceAll(".csv", "_forecast.csv")}');
  }
}

// --- The Main Entry Point ---

void main() {
  // 1. Create a new instance of the forecaster.
  final forecaster = SilverForecaster(
    inputCsvPath: 'lib/datasets/silver.csv',
    outputCsvPath: 'D:/LokalAI/multiagent/llm_finetuner/evaluation_results.csv',
    epochs: 5, // You can configure all hyperparameters here
    numModels: 4,
    hiddenSize: 28,
  );

  // 2. Run the entire process.
  forecaster.run();
}