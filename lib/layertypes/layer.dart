import '../autogradEngine/tensor.dart';

/// The abstract base class for all neural network layers.
///
/// A `Layer` is the fundamental, callable building block of a neural network. It
/// encapsulates both a state (its trainable `parameters`, like weights and biases)
/// and a transformation from input tensors to output tensors (the `forward` pass).
///
/// Layers are designed to be chained together in a model, such as an `SNetwork`,
/// where the output of one layer becomes the input to the next, forming the
/// complete network architecture.
///
/// ### Lifecycle:
/// 1. **Instantiation:** A layer is created (e.g., `DenseLayer(64)`). Its weights are not yet created.
/// 2. **Build:** The first time the layer is called with an input, the `build` method runs
///    automatically. It uses the input's shape to initialize the weights and biases with
///    the correct dimensions. This is called deferred initialization.
/// 3. **Forward Pass:** On every call, the `forward` method is executed to perform the
///    layer's core mathematical operations.
///
/// ### Example
/// ```dart
/// // Define a layer (weights are not created yet).
/// Layer dense = DenseLayer(32, activation: ReLU());
///
/// // Create an input tensor.
/// Tensor<Vector> input = Tensor<Vector>([1.0, 2.0, 3.0]);
///
/// // Call the layer. This will first build it, then run the forward pass.
/// Tensor<Vector> output = dense.call(input) as Tensor<Vector>;
/// ```
abstract class Layer {
  /// A list of all trainable tensors (weights and biases) in the layer.
  ///
  /// This getter is used by the optimizer to know which tensors to update
  /// during training. Layers without parameters (like activation layers)
  /// return an empty list.
  List<Tensor> get parameters;

  /// A user-friendly name for the layer (e.g., 'dense', 'lstm').
  String get name;

  bool _built = false;

  /// Initializes the layer's parameters based on the shape of the first input.
  ///
  /// Subclasses should override this method to create their weights and biases.
  /// This method is called automatically by `call` and should not be called directly.
  void build(Tensor<dynamic> input) {
    _built = true;
  }

  /// The core logic of the layer's transformation.
  ///
  /// Subclasses **must** implement this method to define how they process
  /// input tensors and return an output tensor.
  Tensor<dynamic> forward(Tensor<dynamic> input);

  /// The public, callable interface for the layer.
  ///
  /// When a layer instance is called like a function, this method is executed.
  /// It acts as a wrapper that automatically handles the `build` step on the
  /// first run before executing the main `forward` pass logic.
  Tensor<dynamic> call(Tensor<dynamic> input) {
    if (!_built) {
      build(input);
    }
    return forward(input);
  }
}
