import 'package:flutter_ml/full_library.dart';
import 'package:flutter_ml/gpu_version/SNetworkGPU.dart'; // Adjust to your actual import path

void main() async{
  await CudaEngine.initialize(debug: false);

  List<double> rawX = <double>[
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  ];

  List<double> rawY = <double>[
    0.0,
    1.0,
    1.0,
    0.0
  ];

  SNetworkGPU net = SNetworkGPU();

  net.add(DenseReluTL(8));
  net.add(DenseTL(1));

  net.compile(<int>[4, 2], <int>[4, 1], 0.1);

  print('--- Starting Training ---');

  for (int epoch = 1; epoch <= 10000; epoch = epoch + 1) {
    double currentLoss = net.trainStep(rawX, rawY);

    if (epoch % 100 == 0) {
      print('Epoch $epoch | Loss: ${currentLoss.toStringAsFixed(6)}');
    }
  }

  print('\n--- Final Predictions ---');
  List<dynamic> predictions = net.predict(rawX);

  print('[0, 0] -> Target: 0.0 | Output: ${predictions[0][0].toStringAsFixed(4)}');
  print('[0, 1] -> Target: 1.0 | Output: ${predictions[1][0].toStringAsFixed(4)}');
  print('[1, 0] -> Target: 1.0 | Output: ${predictions[2][0].toStringAsFixed(4)}');
  print('[1, 1] -> Target: 0.0 | Output: ${predictions[3][0].toStringAsFixed(4)}');

  // 8. Prevent Memory Leaks
  net.free();
  CudaEngine.dispose();
}