import 'dart:io';
import 'dart:math';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import '../optimizers/sgd.dart';
import 'denseLayer.dart';
import 'layer.dart';
import 'lstmLayer.dart';

/// A Multi-Timeline Long Short-Term Memory (MT-LSTM) layer.
///
/// This is a custom, hierarchical recurrent layer designed to capture dependencies
/// across multiple, distinct timescales within a single sequence.
///
/// It operates on two levels:
/// 1.  A **Lower Tier** that processes the input at every single timestep, capturing
///     high-frequency, short-term patterns.
/// 2.  A **Higher Tier** that runs at a slower "clock speed." It only updates
///     after a block of lower-tier steps, allowing it to learn low-frequency,
///     long-term trends by processing aggregated information.
///
/// A key feature is the **feedback mechanism**, where the state of the higher-tier
/// memory is fed back as a context to the lower tier at every step. This allows
/// the long-term trend to influence the processing of short-term data.
class DualLSTMLayer extends Layer {
  @override
  String name = 'duallstm';

  final int hiddenSize;
  final int lowerTierClockCycle;

  // --- Parameters for the Lower Tier (e.g., Daily) ---
  late Tensor<Matrix> lW_f, lW_i, lW_c, lW_o;
  late Tensor<Vector> lb_f, lb_i, lb_c, lb_o;

  // --- Parameters for the Higher Tier (e.g., Weekly) ---
  late Tensor<Matrix> hW_f, hW_i, hW_c, hW_o;
  late Tensor<Vector> hb_f, hb_i, hb_c, hb_o;

  DualLSTMLayer(this.hiddenSize, {this.lowerTierClockCycle = 7});

  @override
  List<Tensor> get parameters => [
    lW_f, lb_f, lW_i, lb_i, lW_c, lb_c, lW_o, lb_o,
    hW_f, hb_f, hW_i, hb_i, hW_c, hb_c, hW_o, hb_o,
  ];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    // The lower tier's input is [h_lower, h_cell_higher, x_t]
    int lowerCombinedSize = hiddenSize + hiddenSize + inputSize;

    // The higher tier's input is [h_higher, h_lower]
    int higherCombinedSize = hiddenSize + hiddenSize;

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

    // Initialize lower tier weights
    lW_f = initWeights(lowerCombinedSize, hiddenSize);
    lW_i = initWeights(lowerCombinedSize, hiddenSize);
    lW_c = initWeights(lowerCombinedSize, hiddenSize);
    lW_o = initWeights(lowerCombinedSize, hiddenSize);
    lb_f = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    lb_i = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    lb_c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    lb_o = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    // Initialize higher tier weights
    hW_f = initWeights(higherCombinedSize, hiddenSize);
    hW_i = initWeights(higherCombinedSize, hiddenSize);
    hW_c = initWeights(higherCombinedSize, hiddenSize);
    hW_o = initWeights(higherCombinedSize, hiddenSize);
    hb_f = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    hb_i = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    hb_c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    hb_o = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    int totalSteps = sequence.length;

    // Initialize all states with zeros.
    // 'l' prefix for lower tier, 'h' for higher tier.
    Tensor<Vector> lh = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)); // Lower hidden state
    Tensor<Vector> lc = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)); // Lower cell state
    Tensor<Vector> hh = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)); // Higher hidden state
    Tensor<Vector> hc = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)); // Higher cell state

    for (int i = 0; i < totalSteps; i++) {
      Vector timestep_x_list = sequence[i];
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);

      // --- 1. LOWER TIER UPDATE (runs at every step) ---

      // Feedback: Combine lower hidden, HIGHER cell, and current input
      Tensor<Vector> temp_combined = concatenate(lh, hc);
      Tensor<Vector> combined_input_lower = concatenate(temp_combined, x_t);

      // Forget Gate
      Tensor<Vector> lf_t_linear = matVecMul(lW_f, combined_input_lower);
      Tensor<Vector> lf_t_biased = add(lf_t_linear, lb_f);
      Tensor<Vector> lf_t = sigmoid(lf_t_biased);

      // Input Gate
      Tensor<Vector> li_t_linear = matVecMul(lW_i, combined_input_lower);
      Tensor<Vector> li_t_biased = add(li_t_linear, lb_i);
      Tensor<Vector> li_t = sigmoid(li_t_biased);
      Tensor<Vector> lc_tilde_t_linear = matVecMul(lW_c, combined_input_lower);
      Tensor<Vector> lc_tilde_t_biased = add(lc_tilde_t_linear, lb_c);
      Tensor<Vector> lc_tilde_t = vectorTanh(lc_tilde_t_biased);

      // Cell State Update
      Tensor<Vector> lc_retained = elementWiseMultiply(lf_t, lc);
      Tensor<Vector> lc_new_info = elementWiseMultiply(li_t, lc_tilde_t);
      lc = add(lc_retained, lc_new_info);

      // Output Gate
      Tensor<Vector> lo_t_linear = matVecMul(lW_o, combined_input_lower);
      Tensor<Vector> lo_t_biased = add(lo_t_linear, lb_o);
      Tensor<Vector> lo_t = sigmoid(lo_t_biased);
      Tensor<Vector> lc_activated = vectorTanh(lc);
      lh = elementWiseMultiply(lo_t, lc_activated);

      // --- 2. HIGHER TIER UPDATE (runs periodically) ---
      if (i > 0 && (i + 1) % lowerTierClockCycle == 0) {

        // Input to the higher tier is its own last hidden state (hh)
        // and the aggregated info from the lower tier (the current lh).
        Tensor<Vector> combined_input_higher = concatenate(hh, lh);

        // Forget Gate
        Tensor<Vector> hf_t_linear = matVecMul(hW_f, combined_input_higher);
        Tensor<Vector> hf_t_biased = add(hf_t_linear, hb_f);
        Tensor<Vector> hf_t = sigmoid(hf_t_biased);

        // Input Gate
        Tensor<Vector> hi_t_linear = matVecMul(hW_i, combined_input_higher);
        Tensor<Vector> hi_t_biased = add(hi_t_linear, hb_i);
        Tensor<Vector> hi_t = sigmoid(hi_t_biased);
        Tensor<Vector> hc_tilde_t_linear = matVecMul(hW_c, combined_input_higher);
        Tensor<Vector> hc_tilde_t_biased = add(hc_tilde_t_linear, hb_c);
        Tensor<Vector> hc_tilde_t = vectorTanh(hc_tilde_t_biased);

        // Cell State Update
        Tensor<Vector> hc_retained = elementWiseMultiply(hf_t, hc);
        Tensor<Vector> hc_new_info = elementWiseMultiply(hi_t, hc_tilde_t);
        hc = add(hc_retained, hc_new_info);

        // Output Gate
        Tensor<Vector> ho_t_linear = matVecMul(hW_o, combined_input_higher);
        Tensor<Vector> ho_t_biased = add(ho_t_linear, hb_o);
        Tensor<Vector> ho_t = sigmoid(ho_t_biased);
        Tensor<Vector> hc_activated = vectorTanh(hc);
        hh = elementWiseMultiply(ho_t, hc_activated);
      }
    }

    // Return the final hidden state of the lower, most granular tier.
    return lh;
  }
}

// Assuming the following helper functions/classes exist outside of main:
// Tensor, Layer, MultiLSTMLayer, LSTMLayer, SNetwork, SGD, mse, etc.

void main() {
  // --- Inline Helper Functions ---

  // Inlined prepareRnnData
  void prepareRnnData({
    required List<Tensor<Matrix>> inputs,
    required List<Tensor<Vector>> targets,
    required int numSamples,
    required int sequenceLength,
    required double startOffset,
  }) {
    // Explicit for loop as preferred
    for (int i = 0; i < numSamples; i++) {
      Matrix sequence = <Vector>[];
      double start = startOffset + i * 0.2;
      for (int j = 0; j < sequenceLength; j++) {
        sequence.add(<double>[sin(start + j * 0.1)]);
      }
      inputs.add(Tensor<Matrix>(sequence));
      targets.add(Tensor<Vector>(<double>[sin(start + sequenceLength * 0.1)]));
    }
  }

  // Inlined calculateMSE
  double calculateMSE(SNetwork model, List<Tensor<Matrix>> testX, List<Tensor<Vector>> testY) {
    double totalLoss = 0.0;
    int dataSize = testX.length;

    // Explicit for loop as preferred
    for (int i = 0; i < dataSize; i++) {
      Tensor<Matrix> x = testX[i];
      Tensor<Vector> y_true = testY[i];
      // We use predict() for no gradient tracking on evaluation
      Tensor<Vector> y_pred = model.predict(x) as Tensor<Vector>;
      Tensor<Scalar> loss = mse(y_pred, y_true);
      totalLoss += loss.value;
    }
    return totalLoss / dataSize;
  }

  // Inlined runSingleTrial (Modified for time-based comparison)
  double runSingleTrial({
    required SNetwork model,
    required List<Tensor<Matrix>> trainX,
    required List<Tensor<Vector>> trainY,
    required List<Tensor<Matrix>> testX,
    required List<Tensor<Vector>> testY,
    required int epochs,
    required double learningRate,
    required bool verbose,
    required Duration? timeLimitPerEpoch, // New parameter for time constraint
  }) {
    if (verbose) {
      print('🛠️ Building model parameters...');
    }
    // 1. Build Model (needed to initialize parameters for the optimizer)
    model.call(trainX[0]);

    // 2. Initialize Optimizer
    SGD optimizer = SGD(model.parameters, learningRate: learningRate);

    // 3. Training Loop
    Stopwatch epochStopwatch = Stopwatch();

    // Explicit for loop as preferred
    for (int epoch = 0; epoch < epochs; epoch++) {
      epochStopwatch.start();
      double totalLoss = 0.0;

      // Looping over training data
      for (int i = 0; i < trainX.length; i++) {
        optimizer.zeroGrad();
        Tensor<Scalar> loss = mse(model.call(trainX[i]) as Tensor<Vector>, trainY[i]);
        totalLoss += loss.value;
        loss.backward();
        optimizer.step();
      }
      epochStopwatch.stop();

      // If a time limit is imposed, check if the elapsed time for this faster model
      // has exceeded the slower model's epoch time.
      if (timeLimitPerEpoch != null && epochStopwatch.elapsed > timeLimitPerEpoch) {
        if (verbose) {
          print('🛑 Epoch ${epoch + 1} stopped due to time limit: ${epochStopwatch.elapsedMilliseconds}ms > ${timeLimitPerEpoch.inMilliseconds}ms');
        }
        break; // Stop training this model early
      }

      if (verbose && (epoch + 1) % 5 == 0) {
        double trainLoss = totalLoss / trainX.length;
        double testLoss = calculateMSE(model, testX, testY);
        print('  Epoch ${epoch + 1}/$epochs, Train Loss: ${trainLoss.toStringAsFixed(6)}, Test Loss: ${testLoss.toStringAsFixed(6)} (Time: ${epochStopwatch.elapsedMilliseconds}ms)');
      }

      // Reset stopwatch for the next epoch's timing
      epochStopwatch.reset();
    }

    // 4. Final Evaluation
    double finalTestLoss = calculateMSE(model, testX, testY);
    if (verbose) {
      print('✅ Training finished. Final Test MSE: ${finalTestLoss.toStringAsFixed(8)}');
    }
    return finalTestLoss;
  }

  // --- Global Hyperparameters and Data Setup ---
  int sequenceLength = 14;
  int hiddenSize = 16;
  int epochs = 10;
  double learningRate = 0.1;
  int numTrainSamples = 100;
  int numTestSamples = 20;
  int numTrials = 100;

  // Prepare Data (Only needs to be done once)
  List<Tensor<Matrix>> trainX = <Tensor<Matrix>>[];
  List<Tensor<Vector>> trainY = <Tensor<Vector>>[];
  List<Tensor<Matrix>> testX = <Tensor<Matrix>>[];
  List<Tensor<Vector>> testY = <Tensor<Vector>>[];

  prepareRnnData(inputs: trainX, targets: trainY, numSamples: numTrainSamples, sequenceLength: sequenceLength, startOffset: 0.0);
  prepareRnnData(inputs: testX, targets: testY, numSamples: numTestSamples, sequenceLength: sequenceLength, startOffset: numTrainSamples * 0.2 + 10.0);

  print('📊 Data Prepared. Running ${numTrials} Time-Constrained Trials...');
  print('------------------------------------------------------------------');

  // --- Comparison Loop Variables ---
  int lstmWins = 0;
  int mtLstmWins = 0;
  int ties = 0;
  Duration? mtModelEpochTime;

  // --- Main Comparison Loop ---
  // Explicit for loop as preferred
  for (int trial = 1; trial <= numTrials; trial++) {
    // Models are defined here to keep the structure simple
    SNetwork mtModel = SNetwork(<Layer>[
      DualLSTMLayer(hiddenSize, lowerTierClockCycle: 4),
      DenseLayer(1)
    ]);
    SNetwork stdModel = SNetwork(<Layer>[
      LSTMLayer(hiddenSize),
      DenseLayer(1)
    ]);

    // Determine verbosity
    bool verboseTrial = trial == 1;
    if (verboseTrial) {
      print('\n--- Trial $trial: Determining MT-LSTM Time Limit ---');
    } else if (trial % 10 == 0) {
      // Print status every 10 trials
      stdout.write('\r...Running Trial $trial/$numTrials');
    }

    // --- A. Calibrate MT-LSTM Time (only done for the first trial, then reuse the time) ---
    // If we haven't calibrated the time yet, run a calibration for the MT-LSTM model
    if (mtModelEpochTime == null) {
      // Run the MT-LSTM model for 1 epoch to measure the time
      Stopwatch calibrationStopwatch = Stopwatch()..start();

      // We MUST use the runSingleTrial function to ensure parameters are built and optimizer is set.
      runSingleTrial(
          model: mtModel, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
          epochs: 1, learningRate: learningRate, verbose: false, timeLimitPerEpoch: null
      );

      calibrationStopwatch.stop();
      mtModelEpochTime = calibrationStopwatch.elapsed;

      if (verboseTrial) {
        print('✅ MT-LSTM 1-Epoch Calibration Time: ${mtModelEpochTime!.inMilliseconds}ms');
        print('--- Trial $trial: Time-Constrained Comparison ---');
      }
    }

    // --- B. Run MT-LSTM Trial (Normal Fixed Epochs) ---
    // The MT-LSTM is now our baseline; it runs for the full epochs as its time is our limit.
    double mtLoss = runSingleTrial(
        model: mtModel, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
        epochs: epochs, learningRate: learningRate, verbose: verboseTrial,
        timeLimitPerEpoch: null // No limit for the baseline
    );

    // --- C. Run Standard LSTM Trial (Time-Constrained) ---
    // The Standard LSTM (which is faster per epoch) runs until its total time matches the MT-LSTM's total time.
    double stdLoss = runSingleTrial(
        model: stdModel, trainX: trainX, trainY: trainY, testX: testX, testY: testY,
        epochs: epochs * 2, // Allow it to run more epochs than the MT-LSTM, but constrained by time.
        learningRate: learningRate, verbose: verboseTrial,
        timeLimitPerEpoch: mtModelEpochTime // Use the slower model's time as the limit.
    );

    // Record Wins (using a small epsilon for floating-point comparison)
    double epsilon = 1e-9;
    if (mtLoss < stdLoss - epsilon) {
      mtLstmWins++;
    } else if (stdLoss < mtLoss - epsilon) {
      lstmWins++;
    } else {
      ties++;
    }
  }

  // --- Final Results ---
  print('\r                                                                    '); // Clear the status line
  print('\n\n--- FINAL TIME-CONSTRAINED COMPARISON RESULTS (Over $numTrials Trials) ---');
  print('Baseline (MT-LSTM) Epoch Time: ${mtModelEpochTime!.inMilliseconds}ms');
  print('Standard LSTM was trained until one epoch exceeded this time limit or ${epochs * 2} epochs.');
  print('--------------------------------------------------------------------------------------------');
  print('Standard LSTM Wins: ${lstmWins} times');
  print('MT-LSTM Wins: ${mtLstmWins} times');
  print('Ties (Loss within 1e-9): ${ties} times');

  double lstmWinPercentage = (lstmWins / numTrials) * 100;
  print('\n**Standard LSTM won ${lstmWinPercentage.toStringAsFixed(1)}% of the trials.**');
}