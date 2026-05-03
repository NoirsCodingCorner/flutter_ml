import 'dart:io';
import 'dart:math';

import '../optimizers/sgd.dart';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

/// A Multi-Timeline Long Short-Term Memory (MT-LSTM) layer.
class DualLSTMLayer extends Layer<Matrix, Vector> {
  @override
  String name = 'duallstm';

  int hiddenSize;
  int lowerTierClockCycle;

  // --- Parameters for the Lower Tier (e.g., Daily) ---
  late Tensor<Matrix> lW_f;
  late Tensor<Matrix> lW_i;
  late Tensor<Matrix> lW_c;
  late Tensor<Matrix> lW_o;

  late Tensor<Vector> lb_f;
  late Tensor<Vector> lb_i;
  late Tensor<Vector> lb_c;
  late Tensor<Vector> lb_o;

  // --- Parameters for the Higher Tier (e.g., Weekly) ---
  late Tensor<Matrix> hW_f;
  late Tensor<Matrix> hW_i;
  late Tensor<Matrix> hW_c;
  late Tensor<Matrix> hW_o;

  late Tensor<Vector> hb_f;
  late Tensor<Vector> hb_i;
  late Tensor<Vector> hb_c;
  late Tensor<Vector> hb_o;

  DualLSTMLayer(this.hiddenSize, {this.lowerTierClockCycle = 7});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(lW_f); params.add(lb_f);
    params.add(lW_i); params.add(lb_i);
    params.add(lW_c); params.add(lb_c);
    params.add(lW_o); params.add(lb_o);

    params.add(hW_f); params.add(hb_f);
    params.add(hW_i); params.add(hb_i);
    params.add(hW_c); params.add(hb_c);
    params.add(hW_o); params.add(hb_o);
    return params;
  }

  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int inputSize = 0;
    if (inputMatrix.length > 0) {
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
    lW_f = initWeights(lowerCombinedSize, hiddenSize);
    lW_i = initWeights(lowerCombinedSize, hiddenSize);
    lW_c = initWeights(lowerCombinedSize, hiddenSize);
    lW_o = initWeights(lowerCombinedSize, hiddenSize);
    lb_f = initBias();
    lb_i = initBias();
    lb_c = initBias();
    lb_o = initBias();

    // Initialize higher tier weights
    hW_f = initWeights(higherCombinedSize, hiddenSize);
    hW_i = initWeights(higherCombinedSize, hiddenSize);
    hW_c = initWeights(higherCombinedSize, hiddenSize);
    hW_o = initWeights(higherCombinedSize, hiddenSize);
    hb_f = initBias();
    hb_i = initBias();
    hb_c = initBias();
    hb_o = initBias();

    super.build(input);
  }

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
      Vector timestep_x_list = sequence[i];
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);

      // --- 1. LOWER TIER UPDATE (runs at every step) ---

      // Feedback: Combine lower hidden, HIGHER cell, and current input
      Tensor<Vector> temp_combined = concatenate(lh, hc);
      Tensor<Vector> combined_input_lower = concatenate(temp_combined, x_t);

      // Forget Gate
      Tensor<Vector> lf_t_linear = matVecMul(lW_f, combined_input_lower);
      Tensor<Vector> lf_t_biased = addVector(lf_t_linear, lb_f);
      Tensor<Vector> lf_t = sigmoid(lf_t_biased);

      // Input Gate
      Tensor<Vector> li_t_linear = matVecMul(lW_i, combined_input_lower);
      Tensor<Vector> li_t_biased = addVector(li_t_linear, lb_i);
      Tensor<Vector> li_t = sigmoid(li_t_biased);

      Tensor<Vector> lc_tilde_t_linear = matVecMul(lW_c, combined_input_lower);
      Tensor<Vector> lc_tilde_t_biased = addVector(lc_tilde_t_linear, lb_c);
      Tensor<Vector> lc_tilde_t = vectorTanh(lc_tilde_t_biased);

      // Cell State Update
      Tensor<Vector> lc_retained = elementWiseMultiply(lf_t, lc);
      Tensor<Vector> lc_new_info = elementWiseMultiply(li_t, lc_tilde_t);
      lc = addVector(lc_retained, lc_new_info);

      // Output Gate
      Tensor<Vector> lo_t_linear = matVecMul(lW_o, combined_input_lower);
      Tensor<Vector> lo_t_biased = addVector(lo_t_linear, lb_o);
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
        Tensor<Vector> hf_t_biased = addVector(hf_t_linear, hb_f);
        Tensor<Vector> hf_t = sigmoid(hf_t_biased);

        // Input Gate
        Tensor<Vector> hi_t_linear = matVecMul(hW_i, combined_input_higher);
        Tensor<Vector> hi_t_biased = addVector(hi_t_linear, hb_i);
        Tensor<Vector> hi_t = sigmoid(hi_t_biased);

        Tensor<Vector> hc_tilde_t_linear = matVecMul(hW_c, combined_input_higher);
        Tensor<Vector> hc_tilde_t_biased = addVector(hc_tilde_t_linear, hb_c);
        Tensor<Vector> hc_tilde_t = vectorTanh(hc_tilde_t_biased);

        // Cell State Update
        Tensor<Vector> hc_retained = elementWiseMultiply(hf_t, hc);
        Tensor<Vector> hc_new_info = elementWiseMultiply(hi_t, hc_tilde_t);
        hc = addVector(hc_retained, hc_new_info);

        // Output Gate
        Tensor<Vector> ho_t_linear = matVecMul(hW_o, combined_input_higher);
        Tensor<Vector> ho_t_biased = addVector(ho_t_linear, hb_o);
        Tensor<Vector> ho_t = sigmoid(ho_t_biased);
        Tensor<Vector> hc_activated = vectorTanh(hc);
        hh = elementWiseMultiply(ho_t, hc_activated);
      }
    }

    // Return the final hidden state of the lower, most granular tier.
    return lh;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['lW_f'] = lW_f.value;
    weightsMap['lb_f'] = lb_f.value;
    weightsMap['lW_i'] = lW_i.value;
    weightsMap['lb_i'] = lb_i.value;
    weightsMap['lW_c'] = lW_c.value;
    weightsMap['lb_c'] = lb_c.value;
    weightsMap['lW_o'] = lW_o.value;
    weightsMap['lb_o'] = lb_o.value;

    weightsMap['hW_f'] = hW_f.value;
    weightsMap['hb_f'] = hb_f.value;
    weightsMap['hW_i'] = hW_i.value;
    weightsMap['hb_i'] = hb_i.value;
    weightsMap['hW_c'] = hW_c.value;
    weightsMap['hb_c'] = hb_c.value;
    weightsMap['hW_o'] = hW_o.value;
    weightsMap['hb_o'] = hb_o.value;
    return weightsMap;
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void _copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      int idx = 0;
      for (int i = 0; i < newDataDynamic.length; i = i + 1) {
        List<dynamic> rowDynamic = newDataDynamic[i] as List<dynamic>;
        for (int j = 0; j < rowDynamic.length; j = j + 1) {
          tensor.data[idx] = rowDynamic[j] as double;
          idx = idx + 1;
        }
      }
    }

    void _copyVector(Tensor<Vector> tensor, List<dynamic> newDataDynamic) {
      for (int i = 0; i < newDataDynamic.length; i = i + 1) {
        tensor.data[i] = newDataDynamic[i] as double;
      }
    }

    _copyMatrix(lW_f, weightsMap['lW_f'] as List<dynamic>);
    _copyMatrix(lW_i, weightsMap['lW_i'] as List<dynamic>);
    _copyMatrix(lW_c, weightsMap['lW_c'] as List<dynamic>);
    _copyMatrix(lW_o, weightsMap['lW_o'] as List<dynamic>);
    _copyVector(lb_f, weightsMap['lb_f'] as List<dynamic>);
    _copyVector(lb_i, weightsMap['lb_i'] as List<dynamic>);
    _copyVector(lb_c, weightsMap['lb_c'] as List<dynamic>);
    _copyVector(lb_o, weightsMap['lb_o'] as List<dynamic>);

    _copyMatrix(hW_f, weightsMap['hW_f'] as List<dynamic>);
    _copyMatrix(hW_i, weightsMap['hW_i'] as List<dynamic>);
    _copyMatrix(hW_c, weightsMap['hW_c'] as List<dynamic>);
    _copyMatrix(hW_o, weightsMap['hW_o'] as List<dynamic>);
    _copyVector(hb_f, weightsMap['hb_f'] as List<dynamic>);
    _copyVector(hb_i, weightsMap['hb_i'] as List<dynamic>);
    _copyVector(hb_c, weightsMap['hb_c'] as List<dynamic>);
    _copyVector(hb_o, weightsMap['hb_o'] as List<dynamic>);
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

      print('Step $i -> Loss: ${loss.value.toStringAsFixed(6)} | Output: $outStr');
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