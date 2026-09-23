

import 'package:flutter_ml/full_library.dart';

void main() {
  // --- 1. Define XOR Dataset ---
  final List<Vector> xorInputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ];

  final List<Vector> xorTargets = [
    [0.0], // 0 XOR 0 = 0
    [1.0], // 0 XOR 1 = 1
    [1.0], // 1 XOR 0 = 1
    [0.0]  // 1 XOR 1 = 0
  ];

  // --- 2. Build the SNetwork (Simple Sequential Model) ---
  final SNetwork model = SNetwork([
    // Hidden Layer (2 -> 2)
    DenseLayer(2, activation: ReLU()),
    // Output Layer (2 -> 1)
    DenseLayer(1, activation: Sigmoid())
  ], name: 'XOR-Net');


  // --- 3. Build & Compile the Network ---
  // This first call to `predict` is necessary to build the layers
  // and initialize their parameters before the optimizer needs them.
  model.predict(Tensor<Vector>(xorInputs[0]));

  // The 'model.parameters' list is now populated.
  final SGD optimizer = SGD(model.parameters, learningRate: 0.01);
  model.compile(configuredOptimizer: optimizer);

  // --- 4. Train the Network ---
  print('Training ${model.name} for 5000 epochs...');
  model.fit(xorInputs, xorTargets, epochs: 5000, debug: true);

  // --- 5. Evaluate and Test ---
  print('\n--- Testing Predictions ---');
  for (int i = 0; i < xorInputs.length; i++) {
    final Tensor<Vector> inputTensor = Tensor<Vector>(xorInputs[i]);
    final Tensor<Vector> prediction = model.predict(inputTensor) as Tensor<Vector>;

    final int target = xorTargets[i][0].toInt();
    final double rawOutput = prediction.value[0];
    final int predictedClass = (rawOutput > 0.5) ? 1 : 0;

    print('Input: ${xorInputs[i]}, Target: $target, Predicted: $predictedClass');
  }
}