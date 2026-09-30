import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/activationFunction.dart';
import 'layer.dart';

/// Applies a standard Elman Recurrent Neural Network (RNN) pass over a sequential 2D [Matrix] tensor.
/// Processes input sequences of shape `[sequenceLength, featureSize]` sequentially across timesteps,
/// updating the recurrent hidden state vector [h] via input-to-hidden projections ([W_xh]),
/// hidden-to-hidden transitions ([W_hh]), and a bias vector ([b_h]), followed by an [activation] function.
/// Returns the final hidden state [Vector] after processing all sequence steps.
class RNN extends Layer<Matrix, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'rnn';

  /// Dimensionality of the recurrent hidden state vector.
  int hiddenSize;

  /// Activation function applied element-wise to the updated hidden state vector at each timestep.
  ActivationFunction<Vector> activation;

  /// Learnable input-to-hidden weight matrix tensor of shape `[hiddenSize, inputSize]`.
  late Tensor<Matrix> W_xh;

  /// Learnable hidden-to-hidden recurrent transition weight matrix tensor of shape `[hiddenSize, hiddenSize]`.
  late Tensor<Matrix> W_hh;

  /// Learnable bias vector tensor of shape `[hiddenSize]`.
  late Tensor<Vector> b_h;

  /// Creates an [RNN] layer with the specified [hiddenSize] and recurrent [activation] function.
  RNN(this.hiddenSize, {required this.activation});

  /// Returns all trainable parameters: [W_xh], [W_hh], and [b_h].
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(W_xh);
    params.add(W_hh);
    params.add(b_h);
    return params;
  }

  /// Allocates and initializes input-to-hidden and hidden-to-hidden weight matrices using Xavier initialization
  /// and sets all elements of the bias vector to 0.0.
  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int inputSize = 0;
    if (inputMatrix.isNotEmpty) {
      inputSize = inputMatrix[0].length;
    }
    Random random = Random();

    double xavierStdDev(int fanIn, int fanOut) {
      return sqrt(2.0 / (fanIn + fanOut));
    }

    double inputToHiddenStdDev = xavierStdDev(inputSize, hiddenSize);
    Matrix wXhValues = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < inputSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * inputToHiddenStdDev);
      }
      wXhValues.add(row);
    }

    double hiddenToHiddenStdDev = xavierStdDev(hiddenSize, hiddenSize);
    Matrix wHhValues = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < hiddenSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * hiddenToHiddenStdDev);
      }
      wHhValues.add(row);
    }

    W_xh = Tensor<Matrix>(wXhValues);
    W_hh = Tensor<Matrix>(wHhValues);

    Vector bHValues = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      bHValues.add(0.0);
    }
    b_h = Tensor<Vector>(bHValues);

    super.build(input);
  }

  /// Executes the recurrent RNN forward pass on the CPU across all sequence timesteps and returns the final hidden state [Vector].
  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    Matrix sequence = input.value;
    int totalSteps = sequence.length;

    Vector hValues = [];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      hValues.add(0.0);
    }
    Tensor<Vector> h = Tensor<Vector>(hValues);

    for (int i = 0; i < totalSteps; i = i + 1) {
      Tensor<Vector> xT = Tensor<Vector>(sequence[i]);

      Tensor<Vector> inputPart = matVecMul(W_xh, xT);
      Tensor<Vector> hiddenPart = matVecMul(W_hh, h);
      Tensor<Vector> combined = addVector(addVector(inputPart, hiddenPart), b_h);

      h = activation.call(combined);
    }

    return h;
  }

  /// Returns [W_xh], [W_hh], and [b_h] values as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['W_xh'] = W_xh.value;
    weightsMap['W_hh'] = W_hh.value;
    weightsMap['b_h'] = b_h.value;
    return weightsMap;
  }

  /// Sets [W_xh], [W_hh], and [b_h] directly into their flat 1D data buffers from a map.
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

    copyMatrix(W_xh, weightsMap['W_xh'] as List<dynamic>);
    copyMatrix(W_hh, weightsMap['W_hh'] as List<dynamic>);
    copyVector(b_h, weightsMap['b_h'] as List<dynamic>);
  }
}
