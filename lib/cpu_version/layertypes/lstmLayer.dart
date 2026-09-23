import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class LSTMLayer extends Layer<Matrix, Vector> {
  @override
  String name = 'lstm';

  int hiddenSize;

  late Tensor<Matrix> W_f;
  late Tensor<Vector> b_f;

  late Tensor<Matrix> W_i;
  late Tensor<Vector> b_i;

  late Tensor<Matrix> W_c;
  late Tensor<Vector> b_c;

  late Tensor<Matrix> W_o;
  late Tensor<Vector> b_o;

  LSTMLayer(this.hiddenSize);

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(W_f); params.add(b_f);
    params.add(W_i); params.add(b_i);
    params.add(W_c); params.add(b_c);
    params.add(W_o); params.add(b_o);
    return params;
  }

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

    W_f = initWeights(combinedSize, hiddenSize);
    W_i = initWeights(combinedSize, hiddenSize);
    W_c = initWeights(combinedSize, hiddenSize);
    W_o = initWeights(combinedSize, hiddenSize);

    b_f = initBias();
    b_i = initBias();
    b_c = initBias();
    b_o = initBias();

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

    Tensor<Vector> h = Tensor<Vector>(zeroVector);
    Tensor<Vector> c = Tensor<Vector>(zeroVector);

    for (int i = 0; i < totalSteps; i = i + 1) {
      Vector timestepXList = sequence[i];
      Tensor<Vector> xT = Tensor<Vector>(timestepXList);
      Tensor<Vector> combinedInput = concatenate(h, xT);

      Tensor<Vector> fTLinear = matVecMul(W_f, combinedInput);
      Tensor<Vector> fTBiased = addVector(fTLinear, b_f);
      Tensor<Vector> fT = sigmoid(fTBiased);

      Tensor<Vector> iTLinear = matVecMul(W_i, combinedInput);
      Tensor<Vector> iTBiased = addVector(iTLinear, b_i);
      Tensor<Vector> iT = sigmoid(iTBiased);

      Tensor<Vector> cTildeTLinear = matVecMul(W_c, combinedInput);
      Tensor<Vector> cTildeTBiased = addVector(cTildeTLinear, b_c);
      Tensor<Vector> cTildeT = vectorTanh(cTildeTBiased);

      Tensor<Vector> cRetained = elementWiseMultiply(fT, c);
      Tensor<Vector> cNewInfo = elementWiseMultiply(iT, cTildeT);
      c = addVector(cRetained, cNewInfo);

      Tensor<Vector> oTLinear = matVecMul(W_o, combinedInput);
      Tensor<Vector> oTBiased = addVector(oTLinear, b_o);
      Tensor<Vector> oT = sigmoid(oTBiased);

      Tensor<Vector> cActivated = vectorTanh(c);
      h = elementWiseMultiply(oT, cActivated);
    }

    return h;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['W_f'] = W_f.value;
    weightsMap['b_f'] = b_f.value;
    weightsMap['W_i'] = W_i.value;
    weightsMap['b_i'] = b_i.value;
    weightsMap['W_c'] = W_c.value;
    weightsMap['b_c'] = b_c.value;
    weightsMap['W_o'] = W_o.value;
    weightsMap['b_o'] = b_o.value;
    return weightsMap;
  }

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

    copyMatrix(W_f, weightsMap['W_f'] as List<dynamic>);
    copyVector(b_f, weightsMap['b_f'] as List<dynamic>);

    copyMatrix(W_i, weightsMap['W_i'] as List<dynamic>);
    copyVector(b_i, weightsMap['b_i'] as List<dynamic>);

    copyMatrix(W_c, weightsMap['W_c'] as List<dynamic>);
    copyVector(b_c, weightsMap['b_c'] as List<dynamic>);

    copyMatrix(W_o, weightsMap['W_o'] as List<dynamic>);
    copyVector(b_o, weightsMap['b_o'] as List<dynamic>);
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
      print('Epoch $epoch -> Loss: ${loss.value.toStringAsFixed(6)}');
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