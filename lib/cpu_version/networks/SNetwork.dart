import 'dart:io';
import 'dart:math';
import 'dart:convert';



import '../../logger.dart';

import '../optimizers/optimizer.dart';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/relu.dart';
import '../activationFuncitons/sigmoid.dart';
import '../layertypes/layer.dart';

/// A sequential model that stacks layers linearly.
///
/// `SNetwork` provides a high-level API for building, training, and evaluating
/// neural networks, similar to Keras's Sequential model. It manages the
/// network's layers, parameters, and the entire training lifecycle.
class SNetwork extends Layer<dynamic, dynamic> {
  @override
  String name;
  List<Layer<dynamic, dynamic>> layers;
  late Optimizer optimizer;

  SNetwork(this.layers, {this.name = 'snetwork'});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> allParams = [];
    for (int i = 0; i < layers.length; i = i + 1) {
      List<Tensor<dynamic>> layerParams = layers[i].parameters;
      for (int j = 0; j < layerParams.length; j = j + 1) {
        allParams.add(layerParams[j]);
      }
    }
    return allParams;
  }

  void compile({required Optimizer configuredOptimizer}) {
    optimizer = configuredOptimizer;
  }

  @override
  Tensor<dynamic> forward(Tensor<dynamic> input) {
    Tensor<dynamic> currentOutput = input;
    for (int i = 0; i < layers.length; i = i + 1) {
      Layer<dynamic, dynamic> layer = layers[i];
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
      Logger.log('--- STARTING TRAINING ---');
    }
    Stopwatch stopwatch = Stopwatch();
    stopwatch.start();

    for (int epoch = 0; epoch < epochs; epoch = epoch + 1) {
      double epochLoss = 0.0;

      for (int i = 0; i < inputs.length; i = i + 1) {
        Tensor<Vector> input = Tensor<Vector>(inputs[i]);
        Tensor<Vector> target = Tensor<Vector>(targets[i]);

        Tensor<Vector> finalOutput = forward(input) as Tensor<Vector>;
        Tensor<Scalar> loss = mse(finalOutput, target);

        epochLoss = epochLoss + loss.value;

        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();

        if (debug) {
          int barWidth = 20;
          double progress = (i + 1) / inputs.length;
          int completed = (progress * barWidth).round();

          String bar = '';
          for (int b = 0; b < completed; b = b + 1) {
            bar = '$bar=';
          }
          bar = '$bar>';
          for (int b = 0; b < (barWidth - completed); b = b + 1) {
            bar = '$bar ';
          }

          int percent = (progress * 100).round();
          stdout.write('\rEpoch ${epoch + 1}/$epochs: [$bar] $percent%');
        }
      }

      if (debug) {
        double avgLoss = epochLoss / inputs.length;

        stdout.write('\rEpoch ${epoch + 1}/$epochs: [====================>] 100%, Avg Loss: ${avgLoss.toStringAsFixed(6)}');

        int logInterval = max(1, (epochs / 10).round());

        bool isLogInterval = (epoch + 1) % logInterval == 0;
        if (averageWeight && isLogInterval) {
          double totalWeightSum = 0.0;
          int totalWeightCount = 0;

          List<Tensor<dynamic>> params = parameters;
          for (int p = 0; p < params.length; p = p + 1) {
            Tensor<dynamic> param = params[p];
            if (param.value is Vector) {
              Vector v = param.value as Vector;
              for (int w = 0; w < v.length; w = w + 1) {
                totalWeightSum = totalWeightSum + v[w].abs();
                totalWeightCount = totalWeightCount + 1;
              }
            } else if (param.value is Matrix) {
              Matrix m = param.value as Matrix;
              for (int r = 0; r < m.length; r = r + 1) {
                for (int c = 0; c < m[r].length; c = c + 1) {
                  totalWeightSum = totalWeightSum + m[r][c].abs();
                  totalWeightCount = totalWeightCount + 1;
                }
              }
            }
          }

          if (totalWeightCount > 0) {
            double avg = totalWeightSum / totalWeightCount;
            stdout.write(', Avg Weight Mag: ${avg.toStringAsFixed(4)}');
          }
        }
        Logger.log('');
      }
    }

    stopwatch.stop();
    if (debug) {
      Logger.log('--- TRAINING FINISHED in ${stopwatch.elapsedMilliseconds}ms ---\n');
    }
  }

  void evaluate(List<List<double>> inputs, List<List<double>> targets) {
    int correctPredictions = 0;
    for (int i = 0; i < inputs.length; i = i + 1) {
      Tensor<Vector> testInput = Tensor<Vector>(inputs[i]);
      Tensor<Vector> pred = predict(testInput) as Tensor<Vector>;
      int result = (pred.value[0] > 0.5) ? 1 : 0;

      if (result == targets[i][0]) {
        correctPredictions = correctPredictions + 1;
      }
    }
    double accuracy = (correctPredictions / inputs.length) * 100.0;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> networkWeights = {};
    int layerIndex = 0;
    for (int i = 0; i < layers.length; i = i + 1) {
      Layer<dynamic, dynamic> layer = layers[i];
      if (layer.parameters.isNotEmpty) {
        networkWeights['layer_$layerIndex'] = layer.getWeights();
        layerIndex = layerIndex + 1;
      }
    }
    return networkWeights;
  }

  @override
  void setWeights(Map<String, dynamic> networkWeights) {
    int layerIndex = 0;
    for (int i = 0; i < layers.length; i = i + 1) {
      Layer<dynamic, dynamic> layer = layers[i];
      if (layer.parameters.isNotEmpty) {
        String key = 'layer_$layerIndex';
        if (networkWeights.containsKey(key)) {
          layer.setWeights(networkWeights[key] as Map<String, dynamic>);
          layerIndex = layerIndex + 1;
        }
      }
    }
  }

  Future<void> save(String filePath) async {
    try {
      Map<String, dynamic> networkWeights = getWeights();
      JsonEncoder encoder = const JsonEncoder.withIndent('  ');
      String jsonString = encoder.convert(networkWeights);

      File file = File(filePath);
      await file.writeAsString(jsonString);
      Logger.log('Network weights saved to $filePath');
    } catch (e) {
      Logger.log('Error saving network: $e');
    }
  }

  Future<void> load(String filePath) async {
    try {
      File file = File(filePath);
      bool exists = await file.exists();
      if (!exists) {
        Logger.log('Error loading network: File not found at $filePath');
        return;
      }
      String jsonString = await file.readAsString();
      Map<String, dynamic> networkWeights = jsonDecode(jsonString);

      setWeights(networkWeights);
      Logger.log('Network weights loaded from $filePath');
    } catch (e) {
      Logger.log('Error loading network: $e');
    }
  }

  void inspectGraph(List<double> inputData, List<double> targetData) {
    Logger.log('\n--- Inspecting Computational Graph ---');

    // 1. Convert raw lists to Tensors
    Tensor<Vector> input = Tensor<Vector>(inputData);
    Tensor<Vector> target = Tensor<Vector>(targetData);

    // 2. Run the forward pass
    Tensor<Vector> finalOutput = forward(input) as Tensor<Vector>;

    // 3. Calculate the loss (to complete the graph)
    Tensor<Scalar> loss = mse(finalOutput, target);

    // 4. Print the graph
    loss.printGraph();

    Logger.log('--------------------------------------\n');
  }
}

Future<void> main() async {
  List<Vector> xorInputs = [];
  xorInputs.add([0.0, 0.0]);
  xorInputs.add([0.0, 1.0]);
  xorInputs.add([1.0, 0.0]);
  xorInputs.add([1.0, 1.0]);

  List<Vector> xorTargets = [];
  xorTargets.add([0.0]);
  xorTargets.add([1.0]);
  xorTargets.add([1.0]);
  xorTargets.add([0.0]);

  List<Layer<dynamic, dynamic>> layers = [];
  // 1. INCREASE HIDDEN NEURONS TO 8
  layers.add(DenseLayer(8, activation: ReLU()));
  layers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork model = SNetwork(layers, name: 'XOR-Net');
  Tensor<Vector> initialInputTensor = Tensor<Vector>(xorInputs[0]);
  model.predict(initialInputTensor);

  // 2. LOWER THE LEARNING RATE TO 0.1
  SGD optimizer = SGD(model.parameters, learningRate: 0.1);
  model.compile(configuredOptimizer: optimizer);

  int epochs = 5000;
  Logger.log('Training ${model.name} for $epochs epochs...');
  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);
  model.inspectGraph(xorInputs[0], xorTargets[0]);

  String modelPath = 'xor_model.json';
  await model.save(modelPath);

  Logger.log('\n--- Loading weights into new model ---');

  List<Layer<dynamic, dynamic>> loadedLayers = [];
  // Ensure the loaded architecture matches exactly
  loadedLayers.add(DenseLayer(8, activation: ReLU()));
  loadedLayers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork loadedModel = SNetwork(loadedLayers, name: 'Loaded-XOR-Net');
  loadedModel.predict(initialInputTensor);
  await loadedModel.load(modelPath);

  Logger.log('\n--- Testing Predictions (from LOADED model) ---');
  int i = 0;
  for (int j = 0; j < xorInputs.length; j = j + 1) {
    Vector input = xorInputs[j];
    Tensor<Vector> inputTensor = Tensor<Vector>(input);
    Tensor<Vector> predictionTensor = loadedModel.predict(inputTensor) as Tensor<Vector>;

    int target = xorTargets[i][0].toInt();
    double rawOutput = predictionTensor.value[0];
    int predictedClass = (rawOutput > 0.5) ? 1 : 0;
    bool isCorrect = (predictedClass == target);

    Logger.log('Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: $isCorrect');
    i = i + 1;
  }
}