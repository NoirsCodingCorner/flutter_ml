import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import 'layer.dart';

/// Applies a Long Short-Term Memory (LSTM) recurrent neural network pass over a sequential 2D [Matrix] tensor.
/// Processes input sequences of shape `[sequenceLength, featureSize]` timestep-by-timestep using standard gating mechanisms
/// (forget gate, input gate, candidate cell state, and output gate) to maintain long-term dependencies.
/// Returns the final hidden state [Vector] after processing all sequence steps.
class LSTMLayer extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'lstm';

  /// Dimensionality of the hidden and cell state vectors.
  int hiddenSize;

  /// Forget gate weight matrix tensor of shape `[hiddenSize, hiddenSize + inputSize]`.
  late Tensor<Matrix> wf;

  /// Forget gate bias vector tensor of shape `[hiddenSize]`.
  late Tensor<Vector> bf;

  /// Input gate weight matrix tensor of shape `[hiddenSize, hiddenSize + inputSize]`.
  late Tensor<Matrix> wi;

  /// Input gate bias vector tensor of shape `[hiddenSize]`.
  late Tensor<Vector> bi;

  /// Candidate cell state weight matrix tensor of shape `[hiddenSize, hiddenSize + inputSize]`.
  late Tensor<Matrix> wc;

  /// Candidate cell state bias vector tensor of shape `[hiddenSize]`.
  late Tensor<Vector> bc;

  /// Output gate weight matrix tensor of shape `[hiddenSize, hiddenSize + inputSize]`.
  late Tensor<Matrix> wo;

  /// Output gate bias vector tensor of shape `[hiddenSize]`.
  late Tensor<Vector> bo;

  /// Creates an [LSTMLayer] with the specified [hiddenSize].
  LSTMLayer(this.hiddenSize);

  /// Returns all trainable weight matrices and bias vectors across all gates.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(wf);
    params.add(bf);
    params.add(wi);
    params.add(bi);
    params.add(wc);
    params.add(bc);
    params.add(wo);
    params.add(bo);
    return params;
  }

  /// Allocates and initializes gate weight matrices using uniform scaling based on fan-in and sets all gate biases to 0.0.
  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int inputSize = 0;
    if (inputMatrix.isNotEmpty) {
      inputSize = inputMatrix[0].length;
    }
    int combinedSize = hiddenSize + inputSize;
    Random random = Random();

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

    wf = initWeights(combinedSize, hiddenSize);
    wi = initWeights(combinedSize, hiddenSize);
    wc = initWeights(combinedSize, hiddenSize);
    wo = initWeights(combinedSize, hiddenSize);

    bf = initBias();
    bi = initBias();
    bc = initBias();
    bo = initBias();

    super.build(input);
  }

  /// Executes the recurrent LSTM forward pass on the CPU across all sequence steps and returns the final hidden state [Vector].
  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    Matrix sequence = input.value;
    int totalSteps = sequence.length;

    Vector zeroVector = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      zeroVector.add(0.0);
    }

    Tensor<Vector> h = Tensor<Vector>(zeroVector);
    Tensor<Vector> c = Tensor<Vector>(zeroVector);

    for (int i = 0; i < totalSteps; i = i + 1) {
      Vector timestepXList = sequence[i];
      Tensor<Vector> xT = Tensor<Vector>(timestepXList);
      Tensor<Vector> combinedInput = concatenate(h, xT);

      Tensor<Vector> fTLinear = matVecMul(wf, combinedInput);
      Tensor<Vector> fTBiased = addVector(fTLinear, bf);
      Tensor<Vector> fT = sigmoid(fTBiased);

      Tensor<Vector> iTLinear = matVecMul(wi, combinedInput);
      Tensor<Vector> iTBiased = addVector(iTLinear, bi);
      Tensor<Vector> iT = sigmoid(iTBiased);

      Tensor<Vector> cTildeTLinear = matVecMul(wc, combinedInput);
      Tensor<Vector> cTildeTBiased = addVector(cTildeTLinear, bc);
      Tensor<Vector> cTildeT = vectorTanh(cTildeTBiased);

      Tensor<Vector> cRetained = elementWiseMultiply(fT, c);
      Tensor<Vector> cNewInfo = elementWiseMultiply(iT, cTildeT);
      c = addVector(cRetained, cNewInfo);

      Tensor<Vector> oTLinear = matVecMul(wo, combinedInput);
      Tensor<Vector> oTBiased = addVector(oTLinear, bo);
      Tensor<Vector> oT = sigmoid(oTBiased);

      Tensor<Vector> cActivated = vectorTanh(c);
      h = elementWiseMultiply(oT, cActivated);
    }

    return h;
  }

  /// Returns all gate weights and biases as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['W_f'] = wf.value;
    weightsMap['b_f'] = bf.value;
    weightsMap['W_i'] = wi.value;
    weightsMap['b_i'] = bi.value;
    weightsMap['W_c'] = wc.value;
    weightsMap['b_c'] = bc.value;
    weightsMap['W_o'] = wo.value;
    weightsMap['b_o'] = bo.value;
    return weightsMap;
  }

  /// Sets all gate weight matrices and bias vectors directly into their flat 1D data buffers from a map.
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

    copyMatrix(wf, weightsMap['W_f'] as List<dynamic>);
    copyVector(bf, weightsMap['b_f'] as List<dynamic>);

    copyMatrix(wi, weightsMap['W_i'] as List<dynamic>);
    copyVector(bi, weightsMap['b_i'] as List<dynamic>);

    copyMatrix(wc, weightsMap['W_c'] as List<dynamic>);
    copyVector(bc, weightsMap['b_c'] as List<dynamic>);

    copyMatrix(wo, weightsMap['W_o'] as List<dynamic>);
    copyVector(bo, weightsMap['b_o'] as List<dynamic>);
  }
}

/*void main() {
  print('--- LSTMLayer Isolated Unit Test ---');

  int hiddenSize = 8;
  int sequenceLength = 10;

  // 1. Create the LSTM Layer
  LSTMLayer lstmLayer = LSTMLayer(hiddenSize);

  // 2. Create Dummy Sequential Data
  // A sequence of 10 steps, each with 1 feature
  Matrix inputSequence = [];
  for (int i = 0; i < sequenceLength; i = i + 1) {
    Vector stepFeatures = [];
    stepFeatures.add(i * 0.1);
    inputSequence.add(stepFeatures);
  }
  Tensor<Matrix> input = Tensor<Matrix>(inputSequence);

  // 3. Define a Target Hidden State
  // The LSTM outputs a Vector of size hiddenSize
  Vector targetVector = [];
  for (int i = 0; i < hiddenSize; i = i + 1) {
    targetVector.add(0.5);
  }
  Tensor<Vector> target = Tensor<Vector>(targetVector);

  // 4. Build the layer (Initializes weights and biases)
  lstmLayer.build(input);

  // 5. Setup Optimizer
  SGD optimizer = SGD(lstmLayer.parameters, learningRate: 0.1);

  print('\nStarting Training Loop...');

  // 6. Run a simple training loop to ensure convergence
  int epochs = 1000;
  for (int epoch = 0; epoch < epochs; epoch = epoch + 1) {
    // Forward Pass: Processes the full sequence
    Tensor<Vector> output = lstmLayer.forward(input);

    // Calculate MSE Loss
    Tensor<Scalar> loss = mse(output, target);

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print('Epoch $epoch -> Loss:${loss.value.toStringAsFixed(6)}');
    }

    // Backward Pass: Computes gradients via backpropagation through time
    loss.backward();

    // Update Weights: Uses the flat 1D data array update
    optimizer.step();

    // Zero Gradients for next epoch
    optimizer.zeroGrad();
  }

  print('\nFinal Output (should be near 0.5):');
  print(lstmLayer.forward(input).value);
}*/
