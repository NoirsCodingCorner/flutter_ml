import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A simple Recurrent Neural Network (RNN) layer.
///
/// An `RNN` is the fundamental building block for processing sequential data. It
/// maintains a hidden state (or "memory") that is updated at each timestep by
/// combining the current input with the hidden state from the previous step.
/// This recurrent connection allows it to learn patterns over time.
///
/// The core operation at each timestep `t` is defined by the formula:
/// `$h_t = \text{activation}(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$
///
/// - **Input:** A `Tensor<Matrix>` representing the sequence, with a shape of
///   `[sequence_length, input_size]`.
/// - **Output:** A `Tensor<Vector>` representing the **final** hidden state after
///   processing the entire sequence, with a shape of `[hidden_size]`.
///
/// ### Example
/// ```dart
/// // An RNN layer with 16 memory units and a Tanh activation.
/// Layer rnn = RNN(16, activation: Tanh());
///
/// // An input sequence with 3 timesteps and 5 features each.
/// Tensor<Matrix> sequence = Tensor<Matrix>([
///   [0.1, 0.2, 0.3, 0.4, 0.5],
///   [0.6, 0.7, 0.8, 0.9, 1.0],
///   [0.5, 0.4, 0.3, 0.2, 0.1],
/// ]);
///
/// // The output is the final hidden state vector of length 16.
/// Tensor<Vector> finalState = rnn.call(sequence) as Tensor<Vector>;
/// ```
class RNN extends Layer {
  @override
  String name = 'rnn';

  /// The number of units in the hidden state, representing the "memory" capacity.
  int hiddenSize;

  /// The non-linear activation function to apply to the hidden state.
  /// `Tanh` is the traditional choice for simple RNNs.
  ActivationFunction activation;

  /// The input-to-hidden weight matrix.
  late Tensor<Matrix> W_xh;

  /// The hidden-to-hidden (recurrent) weight matrix.
  late Tensor<Matrix> W_hh;

  /// The bias for the hidden state.
  late Tensor<Vector> b_h;

  RNN(this.hiddenSize, {required this.activation});

  /// Provides the three trainable parameters of the RNN to the optimizer.
  @override
  List<Tensor> get parameters => [W_xh, W_hh, b_h];

  /// Initializes the `W_xh`, `W_hh`, and `b_h` parameter tensors.
  ///
  /// This method infers the `inputSize` from the input sequence and creates
  /// the weight matrices with the correct shapes. It uses Xavier/Glorot
  /// initialization, which is a good practice for layers with `Tanh` activations.
  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    double xavierStdDev(int fanIn, int fanOut) => sqrt(2.0 / (fanIn + fanOut));

    double inputToHiddenStdDev = xavierStdDev(inputSize, hiddenSize);
    Matrix wXhValues = [];
    for (int i = 0; i < hiddenSize; i++) {
      Vector row = [];
      for (int j = 0; j < inputSize; j++) {
        row.add((random.nextDouble() * 2 - 1) * inputToHiddenStdDev);
      }
      wXhValues.add(row);
    }

    double hiddenToHiddenStdDev = xavierStdDev(hiddenSize, hiddenSize);
    Matrix wHhValues = [];
    for (int i = 0; i < hiddenSize; i++) {
      Vector row = [];
      for (int j = 0; j < hiddenSize; j++) {
        row.add((random.nextDouble() * 2 - 1) * hiddenToHiddenStdDev);
      }
      wHhValues.add(row);
    }

    W_xh = Tensor<Matrix>(wXhValues);
    W_hh = Tensor<Matrix>(wHhValues);
    b_h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    super.build(input);
  }

  /// Performs the forward pass for the RNN layer.
  ///
  /// It initializes a zero-vector for the hidden state `h`, then iterates
  /// through each timestep of the input sequence, updating `h` at each step
  /// according to the RNN recurrence relation.
  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    Tensor<Vector> h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Vector timestep_x_list in sequence) {
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);

      Tensor<Vector> inputPart = matVecMul(W_xh, x_t);
      Tensor<Vector> hiddenPart = matVecMul(W_hh, h);
      Tensor<Vector> combined = add(add(inputPart, hiddenPart), b_h);

      h = activation.call(combined) as Tensor<Vector>;
    }

    return h;
  }
}