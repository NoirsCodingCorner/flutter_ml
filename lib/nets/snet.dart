import 'dart:io';
import 'dart:math';
import 'dart:convert'; // <-- ADDED FOR JSON

// Note: Assuming file paths based on previous context
import '../activationFunctions/relu.dart';
import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';
import '../optimizers/optimizers.dart';
import '../optimizers/sgd.dart'; // Assuming SGD path
import '../layertypes/denseLayer.dart'; // Assuming DenseLayer path

/// A sequential model that stacks layers linearly.
///
/// `SNetwork` provides a high-level API for building, training, and evaluating
/// neural networks, similar to Keras's Sequential model. It manages the
/// network's layers, parameters, and the entire training lifecycle.
///
/// ... (rest of docs) ...
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
  }

  void evaluate(List<List<double>> inputs, List<List<double>> targets) {
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

  // ---
  // --- NEW METHODS FOR SAVING/LOADING ---
  // ---

  /// Retrieves the weights of all child layers as a JSON-serializable Map.
  ///
  /// This implements the `Layer` abstract method as a composite.
  /// The map will have keys like 'layer_0', 'layer_1' for layers with parameters.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> networkWeights = {};
    int layerIndex = 0;
    for (Layer layer in layers) {
      // Only save weights from layers that actually have them
      if (layer.parameters.isNotEmpty) {
        // We'll use a key like 'layer_0', 'layer_1', etc.
        networkWeights['layer_$layerIndex'] = layer.getWeights();
        layerIndex++;
      }
    }
    return networkWeights;
  }

  /// Sets the weights of all child layers from a Map.
  ///
  /// This implements the `Layer` abstract method as a composite.
  /// It expects `networkWeights` to have keys like 'layer_0', 'layer_1'.
  @override
  void setWeights(Map<String, dynamic> networkWeights) {
    int layerIndex = 0;
    for (Layer layer in layers) {
      // Only load weights into layers that have them
      if (layer.parameters.isNotEmpty) {
        String key = 'layer_$layerIndex';
        if (networkWeights.containsKey(key)) {
          layer.setWeights(networkWeights[key] as Map<String, dynamic>);
          layerIndex++;
        }
      }
    }
  }

  /// Saves the entire network's weights to a JSON file.
  ///
  /// This is the high-level method you should call.
  /// [filePath] The path to save the .json file (e.g., 'xor_model.json').
  Future<void> save(String filePath) async {
    try {
      // Get the weights from all layers using the getWeights method
      Map<String, dynamic> networkWeights = this.getWeights();

      // --- THIS IS THE FIX ---
      // Create an encoder that adds indentation (2 spaces)
      JsonEncoder encoder = JsonEncoder.withIndent('  ');
      // Convert the map to a pretty-printed string
      String jsonString = encoder.convert(networkWeights);
      // --- END OF FIX ---

      // Write to a file
      File file = File(filePath);
      await file.writeAsString(jsonString);
      print('Network weights saved to $filePath');
    } catch (e) {
      print('Error saving network: $e');
    }
  }

  /// Loads the entire network's weights from a JSON file.
  ///
  /// CRITICAL: The network architecture in code must *exactly*
  /// match the architecture that was saved. This method *requires*
  /// the model to be built (i.e., by calling `predict` once) before
  /// you can load weights into it.
  Future<void> load(String filePath) async {
    try {
      // 1. Read the JSON file
      File file = File(filePath);
      if (!await file.exists()) {
        print('Error loading network: File not found at $filePath');
        return;
      }
      String jsonString = await file.readAsString();

      // 2. Decode the JSON
      Map<String, dynamic> networkWeights = jsonDecode(jsonString);

      // 3. Load the weights back into the model
      this.setWeights(networkWeights);
      print('Network weights loaded from $filePath');
    } catch (e) {
      print('Error loading network: $e');
    }
  }
}

// NOTE: main() must be async to use await
Future<void> main() async {
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
  layers.add(DenseLayer(2, activation: ReLU()));
  layers.add(DenseLayer(1, activation: Sigmoid()));

  final SNetwork model = SNetwork(layers, name: 'XOR-Net');
  final Tensor<Vector> initialInputTensor = Tensor<Vector>(xorInputs[0]);
  model.predict(initialInputTensor); // Build the model

  // --- 3. Compile the Network ---
  final SGD optimizer = SGD(model.parameters, learningRate: 0.01);
  model.compile(configuredOptimizer: optimizer);

  // --- 4. Train the Network ---
  final int epochs = 5000;
  print('Training ${model.name} for $epochs epochs...');
  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);

  // --- 5. SAVE THE TRAINED MODEL ---
  String modelPath = 'xor_model.json';
  await model.save(modelPath);

  // --- 6. CREATE NEW MODEL AND LOAD WEIGHTS ---
  print('\n--- Loading weights into new model ---');
  // Define the *exact* same architecture
  final SNetwork loadedModel = SNetwork([
    DenseLayer(2, activation: ReLU()),
    DenseLayer(1, activation: Sigmoid()),
  ], name: 'Loaded-XOR-Net');

  // Build the new model so its weights are initialized
  loadedModel.predict(initialInputTensor);

  // Load the saved weights
  await loadedModel.load(modelPath);

  // --- 7. Evaluate and Test the *LOADED* model ---
  print('\n--- Testing Predictions (from LOADED model) ---');
  int i = 0;
  for (Vector input in xorInputs) {
    final Tensor<Vector> inputTensor = Tensor<Vector>(input);
    // Use the new loadedModel
    final Tensor<Vector> predictionTensor =
    loadedModel.predict(inputTensor) as Tensor<Vector>;

    final int target = xorTargets[i][0].toInt();
    final double rawOutput = predictionTensor.value[0];
    final int predictedClass = (rawOutput > 0.5) ? 1 : 0;

    print(
        'Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: ${predictedClass == target}');
    i = i + 1;
  }
  // Tensor output = loadedModel.predict(initialInputTensor);
  // output.printGraph();
}