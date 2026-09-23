
import '../../full_library.dart';
import 'optimizer.dart'; // Importiere hier deine OptimizerGPU Basisklasse

class AdamGPU extends OptimizerGPU {
  double learningRate;
  double beta1;
  double beta2;
  double epsilon;
  double weightDecay;

  // VRAM-Speicher für die Adam-Momente
  List<GPUTensor> mCaches = <GPUTensor>[];
  List<GPUTensor> vCaches = <GPUTensor>[];

  int _step = 1;

  AdamGPU(
      super.parameters,
      this.learningRate, {
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-7, // 👈 Angehoben für Mobile FP16 Stabilität
        this.weightDecay = 0.0,
      }) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      // Berechne die Gesamtanzahl der Elemente im Tensor
      int totalElements = parameters[i].shape.fold(1, (a, b) => a * b);

      var m = GPUTensor<dynamic>.empty(parameters[i].shape);
      var v = GPUTensor<dynamic>.empty(parameters[i].shape);

      // 👈 ZWINGEND für Android: Den VRAM explizit mit Nullen überschreiben!
      m.pushData(List<double>.filled(totalElements, 0.0));
      v.pushData(List<double>.filled(totalElements, 0.0));

      mCaches.add(m);
      vCaches.add(v);
    }
  }

  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ADAM_UPDATE);
      tape.putString(parameters[i].id);             // Data
      tape.putString('${parameters[i].id}_grad');   // Gradient
      tape.putString(mCaches[i].id);                // Momentum
      tape.putString(vCaches[i].id);                // Velocity
      tape.putFloat(learningRate);
      tape.putFloat(beta1);
      tape.putFloat(beta2);
      tape.putFloat(epsilon);
      tape.putInt(_step);                           // Current Step
      tape.putFloat(weightDecay);
    }
    // Erhöht den Step-Counter für die C++ Bias-Korrektur
    _step = _step + 1;
  }

  @override
  void zeroGrad(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${parameters[i].id}_grad');
    }
  }

  // Wichtig, um VRAM Leaks zu verhindern
  void free() {
    for (int i = 0; i < mCaches.length; i = i + 1) {
      mCaches[i].free();
      vCaches[i].free();
    }
  }
}