## `Activation` Class

Activation functions are a fundamental component of neural networks. Their primary purpose is to introduce **non-linearity** into the model.

Without a non-linear activation function, a neural network—no matter how many layers deep—would behave like a single, simple linear model. This non-linearity is what allows the network to learn complex patterns and relationships in data.

An activation function takes the "weighted sum" from a neuron (the result of `inputs @ weights + bias`) and transforms it into the neuron's final output, which is then fed as input to the next layer.

### `ActivationFunction` API

This is an abstract class that acts as a common interface (or "contract") for all activation functions. By having this common interface, layers like `DenseLayer` can be written to work with any activation function, making the framework modular and easy to extend.

#### Required Method

* **`Tensor<dynamic> call(Tensor<dynamic> input)`**
    * Applies the activation's transformation to the `input` tensor and returns the result.

-----

### Example Showcase

Activation functions are typically not used directly. Instead, you pass an instance of an activation function to a layer's constructor using the `activation:` parameter.

The layer will then automatically apply the activation during its forward pass.

```dart
/*
  This example shows how to build a simple sequential network (SNetwork)
  to solve the XOR problem, highlighting how activation functions are used.
*/
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

  // --- 2. Build the SNetwork ---
  final SNetwork model = SNetwork([
    // The hidden layer uses ReLU as its activation function.
    DenseLayer(2, activation: ReLU()),
    
    // The output layer uses Sigmoid to get a probability.
    DenseLayer(1, activation: Sigmoid())
  ], name: 'XOR-Net');

  // --- 3. Compile the Network ---
  // (Build step required to initialize parameters for the optimizer)
  model.predict(Tensor<Vector>(xorInputs[0]));
  
  final SGD optimizer = SGD(model.parameters, learningRate: 0.1);
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

```

-----

## Available Implementations

Here are the available activation functions and their typical use cases:

* **ReLU** (`ReLU`, `ReLUMatrix`)

    * The Rectified Linear Unit, defined as `$f(x) = \max(0, x)$`. It outputs the input if it's positive and zero otherwise. It is the default and most common activation for all hidden layers due to its computational speed and strong general performance.

* **LeakyReLU** (`LeakyReLU`)

    * A variant of ReLU that has a small negative slope (`alpha`) for negative inputs (e.g., `$f(x) = 0.01x$` if `$x < 0$`). It is used in hidden layers as an alternative to ReLU to prevent the "Dying ReLU" problem, where neurons can get "stuck" outputting zero and stop learning.

* **ELU** (`ELU`, `ELUMatrix`)

    * The Exponential Linear Unit, defined as `$f(x) = \alpha(e^x - 1)$` for negative inputs. It is another alternative to ReLU in hidden layers that can sometimes lead to faster learning by pushing the mean activation closer to zero.

* **Sigmoid** (`Sigmoid`, `SigmoidMatrix`)

    * A function that squashes any real-valued input into a range between 0 and 1. It is the standard activation for the output layer in binary classification problems, where its output can be interpreted as a probability. It is rarely used in hidden layers.

* **Softmax** (`Softmax`, `SoftmaxMatrix`)

    * Converts a vector of numbers into a probability distribution, where all elements in the vector sum to 1. It is the standard activation for the output layer in multi-class classification problems.

* **SiLU / Swish** (`Swish`)

    * The Sigmoid-weighted Linear Unit, defined as `$f(x) = x \cdot \sigma(x)$`. This is a modern activation for hidden layers that often outperforms ReLU on deeper models.

* **Mish** (`Mish`)

    * A smooth, non-monotonic function defined as `$f(x) = x \cdot \tanh(\text{softplus}(x))$`. It is a state-of-the-art activation for hidden layers that can outperform both ReLU and Swish on challenging benchmarks.