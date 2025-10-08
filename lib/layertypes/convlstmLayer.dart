import 'dart:math';

import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import '../optimizers/sgd.dart';
import 'denseLayer.dart';
import 'flattenLayer.dart';
import 'layer.dart';

/// A Convolutional Long Short-Term Memory (ConvLSTM) layer.
///
/// A `ConvLSTMLayer` is designed for spatiotemporal data, such as a sequence
/// of images (a video). It combines the spatial processing of a `Conv2D` layer
/// with the temporal memory of an `LSTMLayer`.
///
/// Instead of using matrix multiplications, its internal gates use convolutions.
/// This means its hidden state and cell state are not vectors, but 2D feature
/// maps that preserve spatial information from one timestep to the next.
///
/// - **Input:** A `Tensor<Tensor3D>` representing the sequence, with a shape of
///   `[sequence_length, height, width]`.
/// - **Output:** A `Tensor<Matrix>` representing the **final** hidden state, which is a
///   feature map of shape `[height, width]`.
///
/// ### Analogy 🌦️
/// A `ConvLSTM` is like a meteorologist watching a weather radar loop. It uses
/// convolutions to see the shape of a storm in each frame and its LSTM logic
/// to track how that shape moves and changes over time to predict its next location.
///
/// ### Example
/// ```dart
/// Layer convLstm = ConvLSTMLayer(16, 3); // 16 hidden filters, 3x3 kernel
/// ```
class ConvLSTMLayer extends Layer {
  @override
  String name = 'conv_lstm';
  int hiddenFilters;
  int kernelSize;

  late Tensor<Matrix> K_xf, K_hf;
  late Tensor<Matrix> K_xi, K_hi;
  late Tensor<Matrix> K_xc, K_hc;
  late Tensor<Matrix> K_xo, K_ho;
  late Tensor<Matrix> b_f, b_i, b_c, b_o;

  ConvLSTMLayer(this.hiddenFilters, this.kernelSize);

  @override
  List<Tensor> get parameters => [
    K_xf, K_hf, b_f,
    K_xi, K_hi, b_i,
    K_xc, K_hc, b_c,
    K_xo, K_ho, b_o,
  ];

  /// Initializes the 8 kernels and 4 biases for the LSTM gates.
  @override
  void build(Tensor<dynamic> input) {
    Tensor3D inputSequence = input.value as Tensor3D;
    int height = inputSequence[0].length;
    int width = inputSequence[0][0].length;
    Random random = Random();

    Tensor<Matrix> initKernel(int size) {
      double stddev = sqrt(1.0 / (size * size));
      Matrix values = [];
      for (int i = 0; i < size; i++) {
        Vector row = [];
        for (int j = 0; j < size; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    // Since 'same' padding is used, the bias shape matches the input frame shape.
    Tensor<Matrix> initBias() {
      Matrix biasValues = [];
      for (int i = 0; i < height; i++) {
        Vector row = List<double>.filled(width, 0.0);
        biasValues.add(row);
      }
      return Tensor<Matrix>(biasValues);
    }

    K_xf = initKernel(kernelSize);
    K_hf = initKernel(kernelSize);
    K_xi = initKernel(kernelSize);
    K_hi = initKernel(kernelSize);
    K_xc = initKernel(kernelSize);
    K_hc = initKernel(kernelSize);
    K_xo = initKernel(kernelSize);
    K_ho = initKernel(kernelSize);

    b_f = initBias();
    b_i = initBias();
    b_c = initBias();
    b_o = initBias();

    super.build(input);
  }

  /// Performs the forward pass by unrolling the LSTM cell through time.
  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor3D sequence = (input as Tensor<Tensor3D>).value;
    int height = sequence[0].length;
    int width = sequence[0][0].length;

    Matrix zeroMatrix = [];
    for (int i = 0; i < height; i++) {
      zeroMatrix.add(List<double>.filled(width, 0.0));
    }
    Tensor<Matrix> h = Tensor<Matrix>(zeroMatrix);
    Tensor<Matrix> c = Tensor<Matrix>(zeroMatrix);

    for (Matrix timestep_x_matrix in sequence) {
      Tensor<Matrix> x_t = Tensor<Matrix>(timestep_x_matrix);

      Tensor<Matrix> f_t_inputConv = conv2d(K_xf, x_t, padding: 'same');
      Tensor<Matrix> f_t_hiddenConv = conv2d(K_hf, h, padding: 'same');
      Tensor<Matrix> f_t_sum = addMatrix(f_t_inputConv, f_t_hiddenConv);
      Tensor<Matrix> f_t_biased = addMatrix(f_t_sum, b_f);
      Tensor<Matrix> f_t = sigmoidMatrix(f_t_biased);

      Tensor<Matrix> i_t_inputConv = conv2d(K_xi, x_t, padding: 'same');
      Tensor<Matrix> i_t_hiddenConv = conv2d(K_hi, h, padding: 'same');
      Tensor<Matrix> i_t_sum = addMatrix(i_t_inputConv, i_t_hiddenConv);
      Tensor<Matrix> i_t_biased = addMatrix(i_t_sum, b_i);
      Tensor<Matrix> i_t = sigmoidMatrix(i_t_biased);

      Tensor<Matrix> c_tilde_t_inputConv = conv2d(K_xc, x_t, padding: 'same');
      Tensor<Matrix> c_tilde_t_hiddenConv = conv2d(K_hc, h, padding: 'same');
      Tensor<Matrix> c_tilde_t_sum = addMatrix(c_tilde_t_inputConv, c_tilde_t_hiddenConv);
      Tensor<Matrix> c_tilde_t_biased = addMatrix(c_tilde_t_sum, b_c);
      Tensor<Matrix> c_tilde_t = tanhMatrix(c_tilde_t_biased);

      Tensor<Matrix> c_retained = elementWiseMultiplyMatrix(f_t, c);
      Tensor<Matrix> c_new_info = elementWiseMultiplyMatrix(i_t, c_tilde_t);
      c = addMatrix(c_retained, c_new_info);

      Tensor<Matrix> o_t_inputConv = conv2d(K_xo, x_t, padding: 'same');
      Tensor<Matrix> o_t_hiddenConv = conv2d(K_ho, h, padding: 'same');
      Tensor<Matrix> o_t_sum = addMatrix(o_t_inputConv, o_t_hiddenConv);
      Tensor<Matrix> o_t_biased = addMatrix(o_t_sum, b_o);
      Tensor<Matrix> o_t = sigmoidMatrix(o_t_biased);

      Tensor<Matrix> c_activated = tanhMatrix(c);
      h = elementWiseMultiplyMatrix(o_t, c_activated);
    }

    return h;
  }
}
/*void main() {
  SNetwork network = SNetwork([
    ConvLSTMLayer(8, 3),
    FlattenLayer(),
    DenseLayer(1, activation: Sigmoid()),
  ]);

  int buildFrameSize = 7;
  Tensor3D dummySequence = [];
  for (int t = 0; t < 2; t++) {
    Matrix frame = [];
    for (int h = 0; h < buildFrameSize; h++) {
      frame.add(List<double>.filled(buildFrameSize, 0.0));
    }
    dummySequence.add(frame);
  }
  Tensor<Tensor3D> dummyInput = Tensor<Tensor3D>(dummySequence);
  network.predict(dummyInput);

  network.compile(
      configuredOptimizer: SGD(network.parameters, learningRate: 0.005)
  );

  int numSamples = 200;
  int sequenceLength = 4;
  int frameSize = 7;
  List<Tensor3D> inputs = [];
  List<Vector> targets = [];
  Random random = Random();

  for (int i = 0; i < numSamples; i++) {
    bool isVertical = random.nextBool();
    int startX = random.nextInt(frameSize - sequenceLength);
    int startY = random.nextInt(frameSize - sequenceLength);

    Tensor3D sequence = [];
    for (int t = 0; t < sequenceLength; t++) {
      Matrix frame = [];
      for (int r = 0; r < frameSize; r++) {
        frame.add(List<double>.filled(frameSize, 0.0));
      }
      int currentX = isVertical ? startX : startX + t;
      int currentY = isVertical ? startY + t : startY;
      frame[currentY][currentX] = 1.0;
      sequence.add(frame);
    }
    inputs.add(sequence);
    targets.add([isVertical ? 1.0 : 0.0]);
  }

  int epochs = 1000;
  Optimizer optimizer = network.optimizer;
  Stopwatch stopwatch = Stopwatch()..start();

  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0;
    for (int i = 0; i < numSamples; i++) {
      Tensor<Tensor3D> inputTensor = Tensor<Tensor3D>(inputs[i]);
      Tensor<Vector> targetTensor = Tensor<Vector>(targets[i]);

      Tensor<Vector> finalOutput = network.predict(inputTensor) as Tensor<Vector>;
      Tensor<Scalar> loss = mse(finalOutput, targetTensor);
      totalLoss += loss.value;

      loss.backward();
      optimizer.step();
      optimizer.zeroGrad();
    }
    if ((epoch + 1) % 40 == 0) {
      print('Epoch ${epoch + 1}/$epochs, Avg Loss: ${totalLoss / numSamples}');
    }
  }

  stopwatch.stop();
  print('--- TRAINING FINISHED in ${stopwatch.elapsedMilliseconds}ms ---\n');

  int correctPredictions = 0;
  for (int i = 0; i < numSamples; i++) {
    Tensor<Tensor3D> testInput = Tensor<Tensor3D>(inputs[i]);
    Tensor<Vector> pred = network.predict(testInput) as Tensor<Vector>;
    int result = (pred.value[0] > 0.5) ? 1 : 0;
    if (result == targets[i][0]) {
      correctPredictions++;
    }
  }
  double accuracy = (correctPredictions / numSamples) * 100;
  print('Final Model Accuracy: ${accuracy.toStringAsFixed(2)}%');

  print('\n--- TESTING A SPECIFIC SEQUENCE ---');
  Tensor3D horizontalSequenceData = [];
  for (int t = 0; t < sequenceLength; t++) {
    Matrix frame = [];
    for (int r = 0; r < frameSize; r++) {
      frame.add(List<double>.filled(frameSize, 0.0));
    }
    frame[1][1+t] = 1.0; // Pixel moves horizontally at row 1
    horizontalSequenceData.add(frame);
  }
  Tensor<Tensor3D> horizontalSequence = Tensor<Tensor3D>(horizontalSequenceData);
  Tensor<Vector> prediction = network.predict(horizontalSequence) as Tensor<Vector>;
  int finalResult = (prediction.value[0] > 0.5) ? 1 : 0;

  print('Prediction for a horizontal sequence (target=0): $finalResult (Raw: ${prediction.value[0].toStringAsFixed(4)})');

  print('\n--- FINAL COMPUTATIONAL GRAPH ---');
  prediction.printGraph();
}
*/