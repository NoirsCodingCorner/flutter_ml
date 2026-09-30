import 'dart:math';


import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import '../layertypes/layer.dart';

/// A Multi-Timeline Long Short-Term Memory (MT-LSTM) layer operating over a 2D [Matrix] sequence.
/// Maintains two separate tiers of LSTM cells running at different temporal granularities:
/// - A lower tier that updates at every sequential timestep, integrating the lower hidden state, current input, and higher cell state.
/// - A higher tier that updates periodically every [lowerTierClockCycle] timesteps, integrating aggregated representations from the lower tier.
/// Returns the final hidden state [Vector] of the lower tier.
class DualLSTMLayer extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'duallstm';

  /// Dimensionality of the hidden and cell states for both lower and higher tiers.
  int hiddenSize;

  /// The clock cycle interval at which the higher tier updates relative to the lower tier.
  int lowerTierClockCycle;

  // --- Parameters for the Lower Tier (e.g., Daily) ---
  /// Lower-tier forget gate weight matrix.
  late Tensor<Matrix> lWf;
  /// Lower-tier input gate weight matrix.
  late Tensor<Matrix> lWi;
  /// Lower-tier candidate cell state weight matrix.
  late Tensor<Matrix> lWc;
  /// Lower-tier output gate weight matrix.
  late Tensor<Matrix> lWo;

  /// Lower-tier forget gate bias vector.
  late Tensor<Vector> lbf;
  /// Lower-tier input gate bias vector.
  late Tensor<Vector> lbi;
  /// Lower-tier candidate cell state bias vector.
  late Tensor<Vector> lbc;
  /// Lower-tier output gate bias vector.
  late Tensor<Vector> lbo;

  // --- Parameters for the Higher Tier (e.g., Weekly) ---
  /// Higher-tier forget gate weight matrix.
  late Tensor<Matrix> hWf;
  /// Higher-tier input gate weight matrix.
  late Tensor<Matrix> hWi;
  /// Higher-tier candidate cell state weight matrix.
  late Tensor<Matrix> hWc;
  /// Higher-tier output gate weight matrix.
  late Tensor<Matrix> hWo;

  /// Higher-tier forget gate bias vector.
  late Tensor<Vector> hbf;
  /// Higher-tier input gate bias vector.
  late Tensor<Vector> hbi;
  /// Higher-tier candidate cell state bias vector.
  late Tensor<Vector> hbc;
  /// Higher-tier output gate bias vector.
  late Tensor<Vector> hbo;

  /// Creates a [DualLSTMLayer] with the given [hiddenSize] and optional [lowerTierClockCycle].
  DualLSTMLayer(this.hiddenSize, {this.lowerTierClockCycle = 7});

  /// Returns all trainable weight matrices and bias vectors across both lower and higher tiers.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(lWf); params.add(lbf);
    params.add(lWi); params.add(lbi);
    params.add(lWc); params.add(lbc);
    params.add(lWo); params.add(lbo);

    params.add(hWf); params.add(hbf);
    params.add(hWi); params.add(hbi);
    params.add(hWc); params.add(hbc);
    params.add(hWo); params.add(hbo);
    return params;
  }

  /// Allocates and initializes weight matrices using uniform scaling based on fan-in and sets all bias vectors to 0.0.
  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int inputSize = 0;
    if (inputMatrix.isNotEmpty) {
      inputSize = inputMatrix[0].length;
    }
    Random random = Random();

    // The lower tier's input is [h_lower, h_cell_higher, x_t]
    int lowerCombinedSize = hiddenSize + hiddenSize + inputSize;

    // The higher tier's input is [h_higher, h_lower]
    int higherCombinedSize = hiddenSize + hiddenSize;

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

    // Initialize lower tier weights
    lWf = initWeights(lowerCombinedSize, hiddenSize);
    lWi = initWeights(lowerCombinedSize, hiddenSize);
    lWc = initWeights(lowerCombinedSize, hiddenSize);
    lWo = initWeights(lowerCombinedSize, hiddenSize);
    lbf = initBias();
    lbi = initBias();
    lbc = initBias();
    lbo = initBias();

    // Initialize higher tier weights
    hWf = initWeights(higherCombinedSize, hiddenSize);
    hWi = initWeights(higherCombinedSize, hiddenSize);
    hWc = initWeights(higherCombinedSize, hiddenSize);
    hWo = initWeights(higherCombinedSize, hiddenSize);
    hbf = initBias();
    hbi = initBias();
    hbc = initBias();
    hbo = initBias();

    super.build(input);
  }

  /// Executes the dual-timeline recurrent forward pass on the CPU across all sequence steps and returns the final lower-tier hidden state [Vector].
  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    Matrix sequence = input.value;
    int totalSteps = sequence.length;

    Vector zeroVector = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      zeroVector.add(0.0);
    }

    // Initialize all states with zeros.
    Tensor<Vector> lh = Tensor<Vector>(zeroVector); // Lower hidden state
    Tensor<Vector> lc = Tensor<Vector>(zeroVector); // Lower cell state
    Tensor<Vector> hh = Tensor<Vector>(zeroVector); // Higher hidden state
    Tensor<Vector> hc = Tensor<Vector>(zeroVector); // Higher cell state

    for (int i = 0; i < totalSteps; i = i + 1) {
      Vector timestepXList = sequence[i];
      Tensor<Vector> xT = Tensor<Vector>(timestepXList);

      // --- 1. LOWER TIER UPDATE (runs at every step) ---

      // Feedback: Combine lower hidden, HIGHER cell, and current input
      Tensor<Vector> tempCombined = concatenate(lh, hc);
      Tensor<Vector> combinedInputLower = concatenate(tempCombined, xT);

      // Forget Gate
      Tensor<Vector> lfTLinear = matVecMul(lWf, combinedInputLower);
      Tensor<Vector> lfTBiased = addVector(lfTLinear, lbf);
      Tensor<Vector> lfT = sigmoid(lfTBiased);

      // Input Gate
      Tensor<Vector> liTLinear = matVecMul(lWi, combinedInputLower);
      Tensor<Vector> liTBiased = addVector(liTLinear, lbi);
      Tensor<Vector> liT = sigmoid(liTBiased);

      Tensor<Vector> lcTildeTLinear = matVecMul(lWc, combinedInputLower);
      Tensor<Vector> lcTildeTBiased = addVector(lcTildeTLinear, lbc);
      Tensor<Vector> lcTildeT = vectorTanh(lcTildeTBiased);

      // Cell State Update
      Tensor<Vector> lcRetained = elementWiseMultiply(lfT, lc);
      Tensor<Vector> lcNewInfo = elementWiseMultiply(liT, lcTildeT);
      lc = addVector(lcRetained, lcNewInfo);

      // Output Gate
      Tensor<Vector> loTLinear = matVecMul(lWo, combinedInputLower);
      Tensor<Vector> loTBiased = addVector(loTLinear, lbo);
      Tensor<Vector> loT = sigmoid(loTBiased);
      Tensor<Vector> lcActivated = vectorTanh(lc);
      lh = elementWiseMultiply(loT, lcActivated);

      // --- 2. HIGHER TIER UPDATE (runs periodically) ---
      if (i > 0 && (i + 1) % lowerTierClockCycle == 0) {

        // Input to the higher tier is its own last hidden state (hh)
        // and the aggregated info from the lower tier (the current lh).
        Tensor<Vector> combinedInputHigher = concatenate(hh, lh);

        // Forget Gate
        Tensor<Vector> hfTLinear = matVecMul(hWf, combinedInputHigher);
        Tensor<Vector> hfTBiased = addVector(hfTLinear, hbf);
        Tensor<Vector> hfT = sigmoid(hfTBiased);

        // Input Gate
        Tensor<Vector> hiTLinear = matVecMul(hWi, combinedInputHigher);
        Tensor<Vector> hiTBiased = addVector(hiTLinear, hbi);
        Tensor<Vector> hiT = sigmoid(hiTBiased);

        Tensor<Vector> hcTildeTLinear = matVecMul(hWc, combinedInputHigher);
        Tensor<Vector> hcTildeTBiased = addVector(hcTildeTLinear, hbc);
        Tensor<Vector> hcTildeT = vectorTanh(hcTildeTBiased);

        // Cell State Update
        Tensor<Vector> hcRetained = elementWiseMultiply(hfT, hc);
        Tensor<Vector> hcNewInfo = elementWiseMultiply(hiT, hcTildeT);
        hc = addVector(hcRetained, hcNewInfo);

        // Output Gate
        Tensor<Vector> hoTLinear = matVecMul(hWo, combinedInputHigher);
        Tensor<Vector> hoTBiased = addVector(hoTLinear, hbo);
        Tensor<Vector> hoT = sigmoid(hoTBiased);
        Tensor<Vector> hcActivated = vectorTanh(hc);
        hh = elementWiseMultiply(hoT, hcActivated);
      }
    }

    // Return the final hidden state of the lower, most granular tier.
    return lh;
  }

  /// Returns all weight matrices and bias vectors across both lower and higher tiers as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['lW_f'] = lWf.value;
    weightsMap['lb_f'] = lbf.value;
    weightsMap['lW_i'] = lWi.value;
    weightsMap['lb_i'] = lbi.value;
    weightsMap['lW_c'] = lWc.value;
    weightsMap['lb_c'] = lbc.value;
    weightsMap['lW_o'] = lWo.value;
    weightsMap['lb_o'] = lbo.value;

    weightsMap['hW_f'] = hWf.value;
    weightsMap['hb_f'] = hbf.value;
    weightsMap['hW_i'] = hWi.value;
    weightsMap['hb_i'] = hbi.value;
    weightsMap['hW_c'] = hWc.value;
    weightsMap['hb_c'] = hbc.value;
    weightsMap['hW_o'] = hWo.value;
    weightsMap['hb_o'] = hbo.value;
    return weightsMap;
  }

  /// Sets all lower-tier and higher-tier weights and biases directly into their flat 1D data buffers from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      int idx = 0;
      for (int i = 0; i < newDataDynamic.length; i = i + 1) {
        List<dynamic> rowDynamic = newDataDynamic[i] as List<dynamic>;
        for (int j = 0; j < rowDynamic.length; j = j + 1) {
          tensor.data[idx] = rowDynamic[j] as double;
          idx = idx + 1;
        }
      }
    }

    void copyVector(Tensor<Vector> tensor, List<dynamic> newDataDynamic) {
      for (int i = 0; i < newDataDynamic.length; i = i + 1) {
        tensor.data[i] = newDataDynamic[i] as double;
      }
    }

    copyMatrix(lWf, weightsMap['lW_f'] as List<dynamic>);
    copyMatrix(lWi, weightsMap['lW_i'] as List<dynamic>);
    copyMatrix(lWc, weightsMap['lW_c'] as List<dynamic>);
    copyMatrix(lWo, weightsMap['lW_o'] as List<dynamic>);
    copyVector(lbf, weightsMap['lb_f'] as List<dynamic>);
    copyVector(lbi, weightsMap['lb_i'] as List<dynamic>);
    copyVector(lbc, weightsMap['lb_c'] as List<dynamic>);
    copyVector(lbo, weightsMap['lb_o'] as List<dynamic>);

    copyMatrix(hWf, weightsMap['hW_f'] as List<dynamic>);
    copyMatrix(hWi, weightsMap['hW_i'] as List<dynamic>);
    copyMatrix(hWc, weightsMap['hW_c'] as List<dynamic>);
    copyMatrix(hWo, weightsMap['hW_o'] as List<dynamic>);
    copyVector(hbf, weightsMap['hb_f'] as List<dynamic>);
    copyVector(hbi, weightsMap['hb_i'] as List<dynamic>);
    copyVector(hbc, weightsMap['hb_c'] as List<dynamic>);
    copyVector(hbo, weightsMap['hb_o'] as List<dynamic>);
  }
}

/*void main() {
  print('--- DualLSTMLayer Isolated Unit Test ---');

  int hiddenSize = 4;
  int sequenceLength = 14;

  // 1. Create the Dual LSTM Layer
  // A lowerTierClockCycle of 7 means the higher tier updates every 7 steps
  DualLSTMLayer lstmLayer = DualLSTMLayer(hiddenSize, lowerTierClockCycle: 7);

  // 2. Create Dummy Sequential Data
  // A sequence of 14 steps, each with 2 features
  Matrix inputSequence = [];
  for (int i = 0; i < sequenceLength; i = i + 1) {
    Vector stepFeatures = [];
    stepFeatures.add(sin(i * 0.5));
    stepFeatures.add(cos(i * 0.5));
    inputSequence.add(stepFeatures);
  }
  Tensor<Matrix> input = Tensor<Matrix>(inputSequence);

  // 3. Define a Target State
  // The layer outputs a hidden state vector equal to `hiddenSize` (4)
  Vector targetVector = [];
  targetVector.add(0.5);
  targetVector.add(-0.5);
  targetVector.add(0.5);
  targetVector.add(-0.5);
  Tensor<Vector> target = Tensor<Vector>(targetVector);

  // 4. Build the layer (this initializes the parameters based on input shape)
  lstmLayer.build(input);

  // 5. Setup Optimizer
  SGD optimizer = SGD(lstmLayer.parameters, learningRate: 0.1);

  print('\nStarting Training Loop...');

  // 6. Run a training loop to overfit to this single sequence
  int steps = 1000;
  for (int i = 0; i < steps; i = i + 1) {
    // Forward Pass
    Tensor<Vector> output = lstmLayer.forward(input);

    // Calculate Loss
    Tensor<Scalar> loss = mse(output, target);

    // Print progress every 10 steps
    if (i % 10 == 0 || i == steps - 1) {
      String outStr = '[';
      for (int j = 0; j < hiddenSize; j = j + 1) {
        outStr = outStr + output.value[j].toStringAsFixed(4);
        if (j < hiddenSize - 1) {
          outStr = outStr + ', ';
        }
      }
      outStr = outStr + ']';

      print('Step $i -> Loss: ${loss.value.toStringAsFixed(6)} \vert{} Output:$outStr');
    }

    // Backward Pass (computes gradients through time)
    loss.backward();

    // Update Weights
    optimizer.step();

    // Zero Gradients
    optimizer.zeroGrad();
  }

  print('\nTarget was:        [0.5000, -0.5000, 0.5000, -0.5000]');
  print('Training complete. If loss decreased smoothly, gradients are flowing successfully through both timelines.');
}*/
