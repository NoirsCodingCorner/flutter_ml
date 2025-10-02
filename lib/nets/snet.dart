import 'dart:io';
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
class SNetwork extends Layer {
  @override
  final String name;
  final List<Layer> layers;
  late Optimizer optimizer;

  SNetwork(this.layers, {this.name = 'snetwork'});

  @override
  List<Tensor> get parameters =>
      layers.expand((layer) => layer.parameters).toList();

  void compile({required Optimizer configuredOptimizer}) {
    optimizer = configuredOptimizer;
  }

  @override
  Tensor<dynamic> forward(Tensor<dynamic> input) {
    Tensor<dynamic> currentOutput = input;
    for (Layer layer in layers) {
      currentOutput = layer.call(currentOutput);
    }
    return currentOutput;
  }

  Tensor<dynamic> predict(Tensor<dynamic> input) {
    return call(input);
  }

  void fit(List<List<double>> inputs, List<List<double>> targets,
      {int epochs = 100, bool averageWeight = false, bool debug = true}) {
    if (debug) {
      print('--- STARTING TRAINING ---');
    }
    Stopwatch stopwatch = Stopwatch()..start();

    for (int epoch = 0; epoch < epochs; epoch++) {
      double epochLoss = 0.0;

      for (int i = 0; i < inputs.length; i++) {
        Tensor<Vector> input = Tensor<Vector>(inputs[i]);
        Tensor<Vector> target = Tensor<Vector>(targets[i]);

        Tensor<Vector> finalOutput = forward(input) as Tensor<Vector>;
        Tensor<Scalar> loss = mse(finalOutput, target);

        epochLoss += loss.value;

        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();

        // --- NEW: Progress Bar Logic ---
        if (debug) {
          int barWidth = 20;
          double progress = (i + 1) / inputs.length;
          int completed = (progress * barWidth).round();
          String bar = '=' * completed + '>' + ' ' * (barWidth - completed);
          int percent = (progress * 100).round();

          // Use stdout.write and carriage return to update the line
          stdout.write('\rEpoch ${epoch + 1}/$epochs: [$bar] $percent%');
        }
      }

      if (debug) {
        double avgLoss = epochLoss / inputs.length;

        // After the progress bar is full, overwrite it with the final loss
        stdout.write('\rEpoch ${epoch + 1}/$epochs: [====================>] 100%, Avg Loss: ${avgLoss.toStringAsFixed(6)}');

        // Calculate the interval once before the loop for efficiency.
        int logInterval = max(1, (epochs / 10).round());

  // Inside the loop, the check is now always safe.
        bool isLogInterval = (epoch + 1) % logInterval == 0;
        if (averageWeight && isLogInterval) {
          // Calculate and print weight magnitude on a new line for clarity
          double totalWeightSum = 0;
          int totalWeightCount = 0;
          for (Tensor param in parameters) {
            if (param.value is Vector) {
              for (double weight in (param.value as Vector)) {
                totalWeightSum += weight.abs();
                totalWeightCount++;
              }
            } else if (param.value is Matrix) {
              for (Vector row in (param.value as Matrix)) {
                for (double weight in row) {
                  totalWeightSum += weight.abs();
                  totalWeightCount++;
                }
              }
            }
          }
          if (totalWeightCount > 0) {
            double avg = totalWeightSum / totalWeightCount;
            stdout.write(', Avg Weight Mag: ${avg.toStringAsFixed(4)}');
          }
        }
        // Print a newline to move to the next epoch's log
        print('');
      }
    }

    stopwatch.stop();
    if (debug) {
      print('--- TRAINING FINISHED in ${stopwatch.elapsedMilliseconds}ms ---\n');
    }
  }  void evaluate(List<List<double>> inputs, List<List<double>> targets) {
    int correctPredictions = 0;
    for (int i = 0; i < inputs.length; i++) {
      Tensor<Vector> testInput = Tensor<Vector>(inputs[i]);
      Tensor<Vector> pred = predict(testInput) as Tensor<Vector>;
      int result = (pred.value[0] > 0.5) ? 1 : 0;

      if (result == targets[i][0]) {
        correctPredictions++;
      }
    }
    double accuracy = (correctPredictions / inputs.length) * 100;
  }
}