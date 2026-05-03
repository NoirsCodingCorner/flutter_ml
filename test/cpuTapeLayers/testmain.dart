import '../../lib/tensor/tensor.dart';
import '../../lib/tensor/tensor_math_cpu.dart';
import '../../lib/tensor/type_Aliases.dart';
import '../../lib/cpu_version/activationFuncitons/relu.dart';
import '../../lib/cpu_version/networks/SNetwork.dart';
import '../../lib/cpu_version/optimizers/sgd.dart';
import '../../lib/cpu_version/activationFuncitons/sigmoid.dart';
import '../../lib/cpu_version/layertypes/denseLayer.dart';
import '../../lib/cpu_version/layertypes/layer.dart';

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
  print('Training ${model.name} for $epochs epochs...');
  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);

  String modelPath = 'xor_model.json';
  await model.save(modelPath);

  print('\n--- Loading weights into new model ---');

  List<Layer<dynamic, dynamic>> loadedLayers = [];
  // Ensure the loaded architecture matches exactly
  loadedLayers.add(DenseLayer(8, activation: ReLU()));
  loadedLayers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork loadedModel = SNetwork(loadedLayers, name: 'Loaded-XOR-Net');
  loadedModel.predict(initialInputTensor);
  await loadedModel.load(modelPath);

  print('\n--- Testing Predictions (from LOADED model) ---');
  int i = 0;
  for (int j = 0; j < xorInputs.length; j = j + 1) {
    Vector input = xorInputs[j];
    Tensor<Vector> inputTensor = Tensor<Vector>(input);
    Tensor<Vector> predictionTensor = loadedModel.predict(inputTensor) as Tensor<Vector>;

    int target = xorTargets[i][0].toInt();
    double rawOutput = predictionTensor.value[0];
    int predictedClass = (rawOutput > 0.5) ? 1 : 0;
    bool isCorrect = (predictedClass == target);

    print('Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: $isCorrect');
    i = i + 1;
  }
  model.inspectGraph(initialInputTensor.value,xorTargets[0]);

}