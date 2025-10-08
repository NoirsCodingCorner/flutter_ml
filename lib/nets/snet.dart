import 'dart:io';
import 'dart:math';

import '../activationFunctions/relu.dart';
import '../activationFunctions/sigmoid.dart';
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
    //Call simply looks if this already has been built, id yes then it calls forward. Otherwise it builds the net
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

void main() {
  // --- 1. Define XOR Dataset ---
  final List<Vector> xorInputs = [];
  xorInputs.add([0.0, 0.0]);
  xorInputs.add([0.0, 1.0]);
  xorInputs.add([1.0, 0.0]);
  xorInputs.add([1.0, 1.0]);

  final List<Vector> xorTargets = [];
  xorTargets.add([0.0]); // 0 XOR 0 = 0
  xorTargets.add([1.0]); // 0 XOR 1 = 1
  xorTargets.add([1.0]); // 1 XOR 0 = 1
  xorTargets.add([0.0]); // 1 XOR 1 = 0

  // --- 2. Build the SNetwork (Simple Sequential Model) ---
  final List<Layer> layers = [];

  // Hidden Layer (2 -> 2)
  final DenseLayer hiddenLayer = DenseLayer(2, activation: ReLU());
  layers.add(hiddenLayer);

  // Output Layer (2 -> 1)
  final DenseLayer outputLayer = DenseLayer(1, activation: Sigmoid());
  layers.add(outputLayer);

  final SNetwork model = SNetwork(layers, name: 'XOR-Net');

  // ************************************************
  // --- IMPORTANT: Initial Predict/Build Call ---
  // This step runs the first forward pass, calling 'build' on all layers,
  // which populates the 'model.parameters' list. This is necessary
  // if the optimizer needs the parameters list before 'fit' starts.
  // We use the first input data point to establish the shape.
  final Tensor<Vector> initialInputTensor = Tensor<Vector>(xorInputs[0]);
  // The result is not used, only the side-effect of calling 'build' is needed.
  model.predict(initialInputTensor);
  // ************************************************

  // --- 3. Compile the Network ---
  // The 'model.parameters' list is now populated because of the 'predict' call.
  final SGD optimizer = SGD(model.parameters, learningRate: 0.001);
  model.compile(configuredOptimizer: optimizer);

  // --- 4. Train the Network ---
  final int epochs = 5000;
  print('Training ${model.name} for $epochs epochs...');

  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);

  // --- 5. Evaluate and Test ---
  print('\n--- Testing Predictions ---');

  int i = 0;
  for (Vector input in xorInputs) {
    final Tensor<Vector> inputTensor = Tensor<Vector>(input);
    final Tensor<Vector> predictionTensor = model.predict(inputTensor) as Tensor<Vector>;

    // Get the target (label)
    final int target = xorTargets[i][0].toInt();

    // Convert output to a binary decision (0 or 1)
    final double rawOutput = predictionTensor.value[0];
    final int predictedClass = (rawOutput > 0.5) ? 1 : 0;

    print('Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: ${predictedClass == target}');

    // Explicitly increment the counter for the targets list
    i = i + 1;
  }
}