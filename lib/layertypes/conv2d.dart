import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';
import '../optimizers/optimizers.dart';
import 'layer.dart';

/// A 2D convolutional layer.
///
/// This layer creates a set of convolutional filters (kernels) that are convolved
/// with the input to produce a stack of output feature maps. It is the primary

/// building block for Convolutional Neural Networks (CNNs).
///
/// This implementation takes a single-channel input (2D Matrix) and produces a
/// multi-channel output (3D Tensor).
///
/// - **Input:** A `Tensor<Matrix>` representing a single-channel image of shape
///   `[height, width]`.
/// - **Output:** A `Tensor<Tensor3D>` representing the output feature maps, with a
///   shape of `[output_channels, new_height, new_width]`.
///
/// ### Example
/// ```dart
/// // A layer that creates 8 feature maps using 3x3 kernels.
/// Layer conv = Conv2DLayer(8, 3, activation: ReLU());
/// ```
class Conv2DLayer extends Layer {
  @override
  String name = 'conv2d_layer';
  int outChannels;
  int kernelSize;
  String padding;
  ActivationFunction? activation;

  late List<Tensor<Matrix>> kernels;
  late Tensor<Vector> biases;

  Conv2DLayer(
      this.outChannels,
      this.kernelSize, {
        this.padding = 'valid',
        this.activation,
      });

  @override
  List<Tensor> get parameters => [...kernels, biases];

  @override
  void build(Tensor<dynamic> input) {
    Random random = Random();
    kernels = [];

    // He initialization for kernels
    for (int i = 0; i < outChannels; i++) {
      Matrix kernelValues = [];
      double stddev = sqrt(2.0 / (kernelSize * kernelSize));
      for (int r = 0; r < kernelSize; r++) {
        Vector row = [];
        for (int c = 0; c < kernelSize; c++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        kernelValues.add(row);
      }
      kernels.add(Tensor<Matrix>(kernelValues));
    }

    biases = Tensor<Vector>(List<double>.filled(outChannels, 0.0));
    super.build(input);
  }

  @override
  Tensor<Tensor3D> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;
    Tensor3D outputChannels = [];

    for (int i = 0; i < outChannels; i++) {
      Tensor<Matrix> featureMap = conv2d(inputMatrix, kernels[i], padding: padding);
      Tensor<Scalar> bias = Tensor<Scalar>(biases.value[i]);
      Tensor<Matrix> biasedMap = addScalarToMatrix(featureMap, bias);
      outputChannels.add(biasedMap.value);
    }

    Tensor<Tensor3D> out = Tensor<Tensor3D>(outputChannels);

    // The backward pass is implicitly handled by the autograd engine through the
    // operations used above (`conv2d`, `addScalarToMatrix`).
    // A more direct Node could be created here for efficiency if needed.

    // Note: Activation would require a 3D-aware activation function.
    // if (activation != null) {
    //   out = activation.call(out) as Tensor<Tensor3D>;
    // }

    return out;
  }
}


/*void main() {
  int kernelSize = 3;
  int numSamples = 500;
  int epochs = 100;
  double learningRate = 0.01;

  Tensor<Matrix> kernel = Tensor<Matrix>([
    [(Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5)],
    [(Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5)],
    [(Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5), (Random().nextDouble() - 0.5)],
  ]);

  List<Tensor<Matrix>> inputs = [];
  List<Tensor<Matrix>> targets = [];
  Random random = Random();

  for (int i = 0; i < numSamples; i++) {
    Matrix patch = [];
    double oneCount = 0;
    for (int r = 0; r < kernelSize; r++) {
      Vector row = [];
      for (int c = 0; c < kernelSize; c++) {
        double bit = random.nextInt(2).toDouble();
        row.add(bit);
        if (bit == 1.0) {
          oneCount++;
        }
      }
      patch.add(row);
    }
    inputs.add(Tensor<Matrix>(patch));
    targets.add(Tensor<Matrix>([[oneCount]]));
  }

  Optimizer optimizer = Adam([kernel], learningRate: learningRate);

  print('--- Starting Training ---');
  for (int epoch = 0; epoch < epochs; epoch++) {
    double epochLoss = 0;
    for (int i = 0; i < numSamples; i++) {
      optimizer.zeroGrad();

      Tensor<Matrix> input = inputs[i];
      Tensor<Matrix> target = targets[i];

      Tensor<Matrix> output = conv2d(input, kernel);
      Tensor<Scalar> loss = mseMatrix(output, target);

      loss.backward();
      optimizer.step();

      epochLoss += loss.value;
    }
    if ((epoch + 1) % 10 == 0) {
      print('Epoch: ${epoch + 1}, Avg Loss: ${epochLoss / numSamples}');
    }
  }

  print('\n--- Training Complete! ---');
  print('Final Optimized Kernel:\n${kernel.value.map((r) => r.map((v) => v.toStringAsFixed(2)))}');
  print('Expected Kernel (approx):\n[[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [1.00, 1.00, 1.00]]');

  print('\n--- Testing the Counting Kernel ---');
  Tensor<Matrix> testPatch1 = Tensor<Matrix>([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
  Tensor<Matrix> testPatch2 = Tensor<Matrix>([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);

  Tensor<Matrix> result1 = conv2d(testPatch1, kernel);
  Tensor<Matrix> result2 = conv2d(testPatch2, kernel);

  print('Count for patch with 3 ones (Target: 3.0): ${result1.value[0][0].toStringAsFixed(4)}');
  print('Count for patch with 6 ones (Target: 6.0): ${result2.value[0][0].toStringAsFixed(4)}');

  result1.printGraph();
}*/