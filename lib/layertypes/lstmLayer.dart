

import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A Long Short-Term Memory (LSTM) layer.
///
/// An `LSTMLayer` is an advanced recurrent layer designed to overcome the
/// short-term memory limitations of a simple `RNN`. It excels at learning
/// **long-term dependencies** in sequential data.
///
/// It achieves this by using a dedicated **cell state (`c`)** for long-term
/// memory, which acts like a conveyor belt where information can travel
/// across many timesteps without being significantly altered.
///
/// A series of **gates** (Forget, Input, and Output) intelligently control
/// the flow of information into and out of this cell state.
///
/// - **Input:** A `Tensor<Matrix>` representing the sequence, with a shape of
///   `[sequence_length, input_size]`.
/// - **Output:** A `Tensor<Vector>` representing the **final** hidden state after
///   processing the entire sequence, with a shape of `[hidden_size]`.
///
/// ### Example
/// ```dart
/// // An LSTM layer with 32 memory units.
/// Layer lstm = LSTMLayer(32);
///
/// // An input sequence with 10 timesteps and 5 features each.
/// Tensor<Matrix> sequence = Tensor<Matrix>(...);
///
/// // The output is the final hidden state vector of length 32.
/// Tensor<Vector> finalState = lstm.call(sequence) as Tensor<Vector>;
/// ```
class LSTMLayer extends Layer {
  @override
  String name = 'lstm';

  /// The number of units in the hidden state and cell state.
  int hiddenSize;

  /// Weights and biases for the Forget Gate.
  late Tensor<Matrix> W_f;
  late Tensor<Vector> b_f;

  /// Weights and biases for the Input Gate.
  late Tensor<Matrix> W_i;
  late Tensor<Vector> b_i;

  /// Weights and biases for the Candidate Cell State.
  late Tensor<Matrix> W_c;
  late Tensor<Vector> b_c;

  /// Weights and biases for the Output Gate.
  late Tensor<Matrix> W_o;
  late Tensor<Vector> b_o;

  LSTMLayer(this.hiddenSize);

  /// Provides all 8 trainable parameter tensors to the optimizer.
  @override
  List<Tensor> get parameters => [W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o];

  /// Initializes all parameter tensors for the four gates.
  ///
  /// This method infers the `inputSize` from the data and creates the weight
  /// matrices, each with a shape of `[hiddenSize, hiddenSize + inputSize]` to
  /// handle the concatenated `[h_prev, x_t]` input. It uses Glorot/Xavier
  /// initialization, a standard practice for LSTMs.
  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    int combinedSize = hiddenSize + inputSize;
    Random random = Random();

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

    W_f = initWeights(combinedSize, hiddenSize);
    W_i = initWeights(combinedSize, hiddenSize);
    W_c = initWeights(combinedSize, hiddenSize);
    W_o = initWeights(combinedSize, hiddenSize);

    b_f = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_i = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_o = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    super.build(input);
  }

  /// Performs the forward pass for the LSTM layer.
  ///
  /// It initializes the hidden state `h` and cell state `c` to zeros, then
  /// iterates through the input sequence. At each timestep, it performs the
  /// four main LSTM operations (Forget, Input, Cell Update, Output) to
  /// update the states before proceeding to the next timestep.
  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    Tensor<Vector> h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    Tensor<Vector> c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Vector timestep_x_list in sequence) {
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);
      Tensor<Vector> combined_input = concatenate(h, x_t);

      // 1. Forget Gate: Decides what old information to discard from the cell state.
      Tensor<Vector> f_t_linear = matVecMul(W_f, combined_input);
      Tensor<Vector> f_t_biased = add(f_t_linear, b_f);
      Tensor<Vector> f_t = sigmoid(f_t_biased);

      // 2. Input Gate: Decides which new information to store in the cell state.
      Tensor<Vector> i_t_linear = matVecMul(W_i, combined_input);
      Tensor<Vector> i_t_biased = add(i_t_linear, b_i);
      Tensor<Vector> i_t = sigmoid(i_t_biased);

      Tensor<Vector> c_tilde_t_linear = matVecMul(W_c, combined_input);
      Tensor<Vector> c_tilde_t_biased = add(c_tilde_t_linear, b_c);
      Tensor<Vector> c_tilde_t = vectorTanh(c_tilde_t_biased);

      // 3. Cell State Update: Forgets old info and adds new candidate info.
      Tensor<Vector> c_retained = elementWiseMultiply(f_t, c);
      Tensor<Vector> c_new_info = elementWiseMultiply(i_t, c_tilde_t);
      c = add(c_retained, c_new_info);

      // 4. Output Gate: Determines the next hidden state from the updated cell state.
      Tensor<Vector> o_t_linear = matVecMul(W_o, combined_input);
      Tensor<Vector> o_t_biased = add(o_t_linear, b_o);
      Tensor<Vector> o_t = sigmoid(o_t_biased);

      Tensor<Vector> c_activated = vectorTanh(c);
      h = elementWiseMultiply(o_t, c_activated);
    }

    return h;
  }
}