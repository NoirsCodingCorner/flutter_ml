import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';
import '../optimizers/optimizers.dart';

/// A sequential model that stacks layers linearly.
///
/// `SNetwork` provides a high-level API for building, training, and evaluating
/// neural networks, similar to Keras's Sequential model. It manages the
/// network's layers, parameters, and the entire training lifecycle.
///
/// ### Example
/// ```dart
/// // 1. Define the model
/// var network = SNetwork([
///   DenseLayer(8),
///   ReLULayer(),
///   DenseLayer(1),
/// ]);
///
/// // 2. Compile the model
/// network.compile(
///   configuredOptimizer: Adam(network.parameters, learningRate: 0.01)
/// );
///
/// // 3. Train and evaluate
/// network.fit(inputs, targets, epochs: 100);
/// network.evaluate(inputs, targets);
/// ```
class SNetwork {
  /// The list of layers that make up the model's architecture.
  final List<Layer> layers;

  /// The optimizer instance configured for training.
  late Optimizer optimizer;

  SNetwork(this.layers);

  /// A getter to automatically collect all trainable parameters from all layers.
  List<Tensor> get parameters => layers.expand((layer) => layer.parameters).toList();

  /// Configures the model for training.
  ///
  /// This method attaches an optimizer to the network.
  void compile({required Optimizer configuredOptimizer}) {
    optimizer = configuredOptimizer;
  }

  /// Runs a forward pass on an input tensor through all layers.
  ///
  /// This is the internal method for prediction logic.
  Tensor<dynamic> forward(Tensor<dynamic> input) {
    Tensor<dynamic> currentOutput = input;
    for (var layer in layers) {
      currentOutput = layer.call(currentOutput);
    }
    return currentOutput;
  }
  /// Generates output predictions for a given input.
  ///
  /// A user-friendly alias for the `forward` method.
  Tensor<dynamic> predict(Tensor<dynamic> input) {
    return forward(input);
  }

  /// Trains the model for a fixed number of epochs on a dataset.
  ///
  /// This method encapsulates the entire training loop.
  ///
  /// - `inputs`: A list of input samples.
  /// - `targets`: A list of corresponding target values.
  /// - `epochs`: The number of times to iterate over the entire dataset.
  /// - `averageWeight`: If true, prints the average magnitude of all model
  ///   parameters at each logging interval for diagnostic purposes.
  void fit(List<List<double>> inputs, List<List<double>> targets, {int epochs = 100, bool averageWeight = false}) {
    print('--- STARTING TRAINING ---');
    final stopwatch = Stopwatch()..start();

    for (int epoch = 0; epoch < epochs; epoch++) {
      for (int i = 0; i < inputs.length; i++) {
        Tensor<Vector> input = Tensor<Vector>(inputs[i]);
        Tensor<Vector> target = Tensor<Vector>(targets[i]);

        // The forward pass is now handled by the generic `forward` method.
        Tensor<Vector> finalOutput = forward(input) as Tensor<Vector>;
        Tensor<Scalar> loss = mse(finalOutput, target); // Assumes MSE loss for now

        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
      }

      // Logging block
      if ((epoch + 1) % (epochs / 10).round() == 0) {
        print('Epoch ${epoch + 1}/$epochs...');

        if (averageWeight) {
          double totalWeightSum = 0;
          int totalWeightCount = 0;

          for (var param in parameters) {
            if (param.value is Vector) {
              for (var weight in (param.value as Vector)) {
                totalWeightSum += weight.abs();
                totalWeightCount++;
              }
            } else if (param.value is Matrix) {
              for (var row in (param.value as Matrix)) {
                for (var weight in row) {
                  totalWeightSum += weight.abs();
                  totalWeightCount++;
                }
              }
            }
          }

          if (totalWeightCount > 0) {
            double avg = totalWeightSum / totalWeightCount;
            print('  - Avg. Weight Magnitude: ${avg.toStringAsFixed(4)}');
          }
        }
      }
    }

    stopwatch.stop();
    print('--- TRAINING FINISHED in ${stopwatch.elapsedMilliseconds}ms ---\n');
  }

  /// Evaluates the model on a test dataset and prints the accuracy.
  void evaluate(List<List<double>> inputs, List<List<double>> targets) {
    print('--- EVALUATING MODEL ACCURACY ---');
    int correctPredictions = 0;
    for (int i = 0; i < inputs.length; i++) {
      Tensor<Vector> testInput = Tensor<Vector>(inputs[i]);
      Tensor<Vector> pred = predict(testInput) as Tensor<Vector>;
      int result = (pred.value[0] > 0.5) ? 1 : 0; // Assumes binary classification

      if (result == targets[i][0]) {
        correctPredictions++;
      }
    }
    double accuracy = (correctPredictions / inputs.length) * 100;
    print('Model Accuracy: ${accuracy.toStringAsFixed(2)}%');
  }
}