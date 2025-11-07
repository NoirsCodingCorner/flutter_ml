## `SNetwork` Class

A sequential model that stacks layers linearly.

`SNetwork` provides a high-level API for building, training, and evaluating neural networks, similar to Keras's Sequential model. It manages the network's layers, parameters, and the entire training lifecycle.

As `SNetwork` itself extends `Layer`, it can be treated as a single layer (for example, nested within another `SNetwork`).

### Quick Example

```dart
// 1. Define the model
var network = SNetwork([
  DenseLayer(8),
  ReLULayer(),
  DenseLayer(1),
]);

// 2. Compile the model
// Note: predict() must be called once to build the layers
network.predict(Tensor<Vector>([0.0, 0.0]));
network.compile(
  configuredOptimizer: Adam(network.parameters, learningRate: 0.01)
);

// 3. Train and evaluate
network.fit(inputs, targets, epochs: 100);
network.evaluate(inputs, targets);
```

-----

### Properties

* **`name`**: `String`
  A user-friendly name for the network (e.g., 'snetwork').

* **`layers`**: `List<Layer>`
  The list of layers that are stacked sequentially to form the model.

* **`parameters`**: `List<Tensor>`
  A getter that automatically collects and returns all trainable `parameters` (weights and biases) from all layers in the `layers` list.

* **`optimizer`**: `Optimizer`
  The optimizer instance (e.g., `SGD`, `Adam`) that will be used to update the model's parameters during training. This is set by the `compile` method.

-----

### Core Methods

#### Setup and Execution

* **`SNetwork(List<Layer> layers, {String name})`**
  The constructor, which takes the list of layers as its primary argument.

* **`void compile({required Optimizer configuredOptimizer})`**
  Assigns the provided `configuredOptimizer` to the network's `optimizer` property. This must be called before training.

* **`Tensor<dynamic> forward(Tensor<dynamic> input)`**
  The core logic of the network. It executes the forward pass by passing the `input` through the first layer, then passing that layer's output to the next, and so on, returning the final output.

* **`Tensor<dynamic> predict(Tensor<dynamic> input)`**
  The public-facing method for running inference. This is an alias for the `call()` method (inherited from `Layer`), which handles the initial `build` step on the first run before executing the `forward` pass.

#### Training and Evaluation

* **`void fit(List<List<double>> inputs, List<List<double>> targets, {int epochs, ...})`**
  The main training loop. It iterates over the provided data for a specified number of `epochs`. In each epoch, it:

    1.  Passes each `input` through the network (`forward`).
    2.  Calculates the loss (hardcoded as MSE) against the `target`.
    3.  Performs backpropagation on the loss (`loss.backward()`).
    4.  Updates all parameters (`optimizer.step()`).
    5.  Resets all gradients (`optimizer.zeroGrad()`).
    6.  Displays a progress bar and average loss if `debug` is true.

* **`void evaluate(List<List<double>> inputs, List<List<double>> targets)`**
  Evaluates the model's performance on a test set. It calculates accuracy for a binary classification task (assuming a 0.5 threshold) but does not return the value.

-----

### Full Example (from `main`)

The `main` function in the file provides a complete, runnable example of instantiating, compiling, training, and testing the `SNetwork` to solve the classic XOR problem.

```dart
void main() {
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

  // Hidden Layer (2 -> 2)
  final DenseLayer hiddenLayer = DenseLayer(2, activation: ReLU());
  layers.add(hiddenLayer);

  // Output Layer (2 -> 1)
  final DenseLayer outputLayer = DenseLayer(1, activation: Sigmoid());
  layers.add(outputLayer);

  final SNetwork model = SNetwork(layers, name: 'XOR-Net');

  // ************************************************
  // --- IMPORTANT: Initial Predict/Build Call ---
  // This step runs the first forward pass, calling 'build' on all layers,
  // which populates the 'model.parameters' list. This is necessary
  // if the optimizer needs the parameters list before 'fit' starts.
  // We use the first input data point to establish the shape.
  final Tensor<Vector> initialInputTensor = Tensor<Vector>(xorInputs[0]);
  // The result is not used, only the side-effect of calling 'build' is needed.
  model.predict(initialInputTensor);
  // ************************************************

  // --- 3. Compile the Network ---
  // The 'model.parameters' list is now populated because of the 'predict' call.
  final SGD optimizer = SGD(model.parameters, learningRate: 0.1);
  model.compile(configuredOptimizer: optimizer);

  // --- 4. Train the Network ---
  final int epochs = 5000;
  print('Training ${model.name} for $epochs epochs...');

  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);

  // --- 5. Evaluate and Test ---
  print('\n--- Testing Predictions ---');

  int i = 0;
  for (Vector input in xorInputs) {
    final Tensor<Vector> inputTensor = Tensor<Vector>(input);
    final Tensor<Vector> predictionTensor = model.predict(inputTensor) as Tensor<Vector>;

    // Get the target (label)
    final int target = xorTargets[i][0].toInt();

    // Convert output to a binary decision (0 or 1)
    final double rawOutput = predictionTensor.value[0];
    final int predictedClass = (rawOutput > 0.5) ? 1 : 0;

    print('Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: ${predictedClass == target}');

    // Explicitly increment the counter for the targets list
    i = i + 1;
  }
}
```