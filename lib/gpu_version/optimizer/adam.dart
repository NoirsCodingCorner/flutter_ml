import '../../full_library.dart';

/// Applies the Adam optimizer to every element of its parameter tensors.
/// Adam combines the advantages of AdaGrad and RMSProp by computing adaptive learning rates
/// for each parameter using the first and second moments of the gradients.
class AdamGPU extends OptimizerGPU {
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

  /// Initializes the Adam optimizer and records moment cache allocations into [allocTape].
  /// Requires the list of trainable [parameters], the [learningRate], and [allocTape].
  /// Optional parameters [beta1], [beta2], [epsilon], and [weightDecay] allow fine-tuning of the algorithm.
  AdamGPU(
    super.parameters,
    super.learningRate,
    AllocBuffer allocTape, {
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-7,
    this.weightDecay = 0.0,
  }) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      GPUTensor<dynamic> m = GPUTensor<dynamic>.empty(
        parameters[i].shape,
        allocTape,
        requiresGrad: false,
        sourceOp: 'AdamGPU:m',
      );
      GPUTensor<dynamic> v = GPUTensor<dynamic>.empty(
        parameters[i].shape,
        allocTape,
        requiresGrad: false,
        sourceOp: 'AdamGPU:v',
      );

      mCaches.add(m);
      vCaches.add(v);
    }
  }

  /// Appends the Adam weight update operations to the provided [tape].
  @override
  void step(CommandBuffer tape) {
    for (int i = 0; i < parameters.length; i = i + 1) {
      adamUpdateGPU(
        parameters[i],
        mCaches[i],
        vCaches[i],
        learningRate,
        beta1,
        beta2,
        epsilon,
        currentStep,
        weightDecay,
        tape,
      );
    }
  }

  /// Patches the compiled tape with the current step and learning rate, then advances [currentStep].
  /// Call this once per training step, right before replaying the optimize tape.
  @override
  void refreshStep(CommandBuffer tape) {
    tape.patchFloat('lr', learningRate);
    tape.patchInt('step', currentStep);
    currentStep = currentStep + 1;
  }
}
