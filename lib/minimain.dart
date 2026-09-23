import '../gpu_version/ffi/cudaEngine.dart';
import 'full_library.dart';
import 'gpu_version/SNetworkGPU.dart';

void main() async {
  // Passe das Target an dein Gerät an
  await GPUEngine.initialize(debug: false, target: Target.cuda);

  SNetworkGPU net = SNetworkGPU();
  net.add(DenseTL(16));
  net.add(ReLULayerMatrixTapeLayer());
  net.add(DenseTL(1));
  net.add(SigmoidMatrixTL());
  // ✅ FIX 1: Batch Size 4! Das Netzwerk verarbeitet alle 4 Samples gleichzeitig.
  net.compile([4, 2], [4, 1], 0.02, useAdam: true);
  // ✅ FIX 2: Daten flach in einem einzigen Batch-Array übergeben
  List<double> xBatch = [
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  ];

  List<double> yBatch = [
    0.0,
    1.0,
    1.0,
    0.0
  ];

  for (int epoch = 1; epoch <= 2000; epoch++) {
    // Ein einziger trainStep pro Epoche!
    // Der Loss ist automatisch der Durchschnitt aller 4 Samples.
    double loss = net.trainStep(xBatch, yBatch);

    if (epoch % 100 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: $loss");
    }
  }

  // ✅ FIX 3: Batch-Vorhersage auf einmal abrufen
  var prediction = net.predict(xBatch);
  print("\nBatch Predictions (Ziel: 0, 1, 1, 0):");
  print(prediction);

  net.free();
  GPUEngine.dispose();
}