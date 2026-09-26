import '../../full_library.dart';

/// Applies the Adam optimizer to every element of its parameter tensors.
/// Adam combines the advantages of AdaGrad and RMSProp by computing adaptive learning rates
/// for each parameter using the first and second moments of the gradients.
class AdamGPU extends OptimizerGPU {
  /// The step size used to update the weights.
  double learningRate;

  /// The exponential decay rate for the first moment estimates (momentum).
  double beta1;

  /// The exponential decay rate for the second moment estimates (variance).
  double beta2;

  /// A very small value added to the denominator to prevent division by zero.
  double epsilon;

  /// The L2 penalty factor applied to the weights to prevent overfitting.
  double weightDecay;

  /// Persistent VRAM caches storing the first moment (moving average of the gradient) for each parameter.
  List<GPUTensor> mCaches = <GPUTensor>[];

  /// Persistent VRAM caches storing the second moment (moving average of the squared gradient) for each parameter.
  List<GPUTensor> vCaches = <GPUTensor>[];

  /// Tracks the current optimization iteration for bias correction.
  /// Exposed publicly so training can be perfectly paused, saved, and resumed.
  int currentStep = 1;

  /// Initializes the Adam optimizer and allocates VRAM for the moment caches.
  /// Requires the list of trainable [parameters] and the [learningRate].
  /// Optional parameters [beta1], [beta2], [epsilon], and [weightDecay] allow fine-tuning of the algorithm.
  AdamGPU(
      super.parameters,
      this.learningRate, {
        this.beta1 = 0.9,
        this.beta2 = 0.999,
        this.epsilon = 1e-7,
        this.weightDecay = 0.0,
      }) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      int totalElements = 1;
      for (int s = 0; s < parameters[i].shape.length; s = s + 1) {
        totalElements = totalElements * parameters[i].shape[s];
      }

      GPUTensor<dynamic> m = GPUTensor<dynamic>.empty(parameters[i].shape);
      GPUTensor<dynamic> v = GPUTensor<dynamic>.empty(parameters[i].shape);

      List<double> zeroData = <double>[];
      for (int e = 0; e < totalElements; e = e + 1) {
        zeroData.add(0.0);
      }

      m.pushData(zeroData);
      v.pushData(zeroData);

      mCaches.add(m);
      vCaches.add(v);
    }
  }

  /// Appends the Adam weight update operations to the provided [tape].
  /// Increments the [currentStep] automatically after processing all parameters.
  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ADAM_UPDATE);
      tape.putString(parameters[i].id);
      tape.putString('${parameters[i].id}_grad');
      tape.putString(mCaches[i].id);
      tape.putString(vCaches[i].id);
      tape.putFloat(learningRate);
      tape.putFloat(beta1);
      tape.putFloat(beta2);
      tape.putFloat(epsilon);
      tape.putInt(currentStep);
      tape.putFloat(weightDecay);
    }
    currentStep = currentStep + 1;
  }

  /// Appends the zero-gradient operations to the [tape] to clear the gradients for the next pass.
  @override
  void zeroGrad(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      tape.putInt(OP_ZERO_GRAD);
      tape.putString('${parameters[i].id}_grad');
    }
  }

  /// Frees VRAM for all persistently allocated moment caches.
  void free() {
    for (int i = 0; i < mCaches.length; i = i + 1) {
      mCaches[i].free();
      vCaches[i].free();
    }
  }
}