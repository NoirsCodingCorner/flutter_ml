# flutter\_ml

[](https://www.google.com/search?q=https://pub.dev/packages/flutter_ml)
[](https://opensource.org/licenses/MIT)

A deep learning library for Dart, built from the ground up with a pure Dart autograd engine.

This project provides a Keras-inspired high-level API (`SNetwork`) for building and training neural networks, powered by a custom automatic differentiation engine written purely in Dart.


## A Library for Learning

This package is designed for learning. To get started, I recommend looking directly into the `Tensor` class. It contains the bare-bones building blocks of the autograd engine and reveals everything about how gradient propagation works.

The entire structure is open. All math operations, layers, and optimizers are written in simple, readable Dart. This makes it incredibly easy to debug, experiment with, and understand how a complex system like this works.

This transparency comes at a performance cost (Currently working on adding cuda support as well as model loading saving) but it makes the engine a perfect tool for understanding *why* things work the way they do.

-----

## Features

This package includes a wide range of features, from the low-level engine to high-level model APIs.

* **Pure Dart Autograd Engine:** Automatically computes gradients for any model.
* **Dynamic Computational Graphs:** Graphs are built on the fly, just like in PyTorch.
* **Familiar Layer API:** A general `Layer` structure inspired by Keras and PyTorch.
* **High-Level Model API:** The `SNetwork` class for simple model stacking, compiling, and training.
* **Detailed Graph Printouts:** A custom logger can print the entire computational graph.
* **Modern Architecture Support:** Includes all building blocks for **Transformer** networks.
* **Comprehensive Components:** A wide array of layers, optimizers, and activation functions are ready to use.

-----

### Full Component List

* **Core Layers**
    * `DenseLayer`: Standard fully-connected layer for `Vector` data.
    * `DenseLayerMatrix`: Batch-processed fully-connected layer for `Matrix` data.
* **Recurrent Layers**
    * `RNN`: A simple Recurrent Neural Network (RNN) layer.
    * `LSTMLayer`: A Long Short-Term Memory (LSTM) recurrent layer.
    * `DualLSTMLayer`: A custom hierarchical LSTM with two internal tiers (fast and slow).
    * `MultiTierLSTMLayer`: A generalized, hierarchical LSTM with an arbitrary number of configured timescales.
    * `GeneralizedChainedScaleLayer`: A self-contained, multi-scale recurrent layer for processing high-frequency data.
* **Convolutional Layers**
    * `Conv2DLayer`: A 2D convolutional layer for 2D `Matrix` inputs.
    * `ConvLSTMLayer`: A Convolutional LSTM layer for spatiotemporal data (like video).
* **Pooling Layers**
    * `MaxPooling1DLayer` / `MaxPooling2DLayer`
    * `AveragePooling2DLayer`
    * `GlobalAveragePoolingLayer` / `GlobalAveragePooling1D`
* **Transformer Layers**
    * `EmbeddingLayer` / `EmbeddingLayerMatrix`: Converts token indices to dense vectors.
    * `PositionalEncoding`: Injects sinusoidal position information.
    * `SingleHeadAttention`: A single head of the self-attention mechanism.
    * `MultiHeadAttention`: Runs multiple `SingleHeadAttention` heads in parallel.
    * `TransformerEncoderBlock`: A full encoder block (Attention + FeedForward).
* **Normalization & Utility Layers**
    * `BatchNorm1D` / `BatchNorm2D`: Batch Normalization for 1D and 3D data.
    * `LayerNormalization` / `LayerNormalizationVector`: Layer Normalization for `Matrix` and `Vector` data.
    * `DropoutLayer` / `DropoutLayerMatrix`: Regularization layer for `Vector` and `Matrix` inputs.
    * `FlattenLayer`: Reshapes a `Matrix` into a `Vector`.
    * `ReshapeVectorToMatrixLayer`: Reshapes a `Vector` to a 1x1 `Matrix`.
* **Activation Layers (as standalone layers)**
    * `ReLULayer` / `ReLULayerMatrix`

#### Available Optimizers

* **`SGD`**: The standard Stochastic Gradient Descent.
* **`Momentum`**: `SGD` with the addition of a momentum (velocity) term.
* **`NAG`**: Nesterov Accelerated Gradient, an improvement on `Momentum`.
* **`Adagrad`**: An adaptive optimizer good for sparse data.
* **`RMSprop`**: An adaptive optimizer that performs well with RNNs.
* **`Adam`**: The most common, general-purpose adaptive optimizer.
* **`AMSGrad`**: A variant of `Adam` that fixes a potential convergence issue.
* **`AdamW`**: A variant of `Adam` that improves weight decay (L2 regularization).


### Available Activation Functions

* **`ReLU`**
* **`LeakyReLU`**
* **`ELU`**
* **`Sigmoid`** (For binary classification outputs)
* **`Softmax`** (For multi-class classification outputs)
* **`SiLU`** / **`Swish`**
* **`Mish`**


-----

## Quick Start: The XOR Example

This example shows the full workflow: defining a model, compiling it, and training it to solve the classic XOR problem.

```dart
/* A complete, runnable example of training a simple network 
  to solve the XOR problem.
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

## Advanced Showcase: Building a Transformer

This engine is powerful enough to run modern architectures. The library includes all the necessary building blocks to create a full Transformer Encoder from scratch.

This example shows how to build a sentiment classifier by stacking the available Transformer components.

```dart
/*
  This code snippet shows how all the building blocks are 
  assembled into a single SNetwork for sentiment analysis.
*/

// --- 1. Define Model Hyperparameters ---
int vocabSize = 15;      // How many unique words in our vocabulary
int dModel = 16;         // The "width" of the model (embedding dimension)
int numHeads = 2;        // Number of attention heads
int dff = 32;            // Hidden dimension of the feed-forward network
int maxSequenceLength = 10; // Max sentence length for positional encoding

// --- 2. Assemble the SNetwork ---
SNetwork sentimentClassifier = SNetwork([
  // 1. Convert word indices to vectors
  EmbeddingLayer(vocabSize, dModel),
  
  // 2. Add word order information
  PositionalEncoding(maxSequenceLength, dModel),
  
  // 3. Run through two Transformer blocks
  TransformerEncoderBlock(dModel, numHeads, dff),
  TransformerEncoderBlock(dModel, numHeads, dff),
  
  // 4. Pool the final sequence into a single vector
  GlobalAveragePooling1D(),
  
  // 5. Classify the vector (0.0 = negative, 1.0 = positive)
  DenseLayer(1, activation: Sigmoid()),
]);

// --- 3. Build, Compile, and Train ---
// (Build the model with a dummy input)
sentimentClassifier.predict(Tensor<Vector>([1, 2, 3])); 

// (Compile with an optimizer)
sentimentClassifier.compile(
    configuredOptimizer: Adam(sentimentClassifier.parameters, learningRate: 0.01)
);

// (Train the model)
// sentimentClassifier.fit(inputs, targets, epochs: 100);
```

-----
## ❗️ Status: Work in Progress

This library is in the early stages of development. It is a large, ongoing project. The API is not yet stable and is subject to change.

The entire engine is written in pure Dart. While this is fantastic for learning and debugging, it is **not yet optimized for performance**. This is an experimentation and learning package, not a production-ready training framework (yet).

-----

## Installation

This package is not yet published on pub.dev. To use it, add it to your `pubspec.yaml` as a git dependency:

```yaml
dependencies:
  flutter_ml:
    git:
      url: https://github.com/your-username/flutter_ml.git
      ref: main
```

## Contributing & Future Roadmap

This project is a long-term effort. Contributions are extremely welcome. If you are interested in machine learning or high-performance Dart, feel free to open an issue or submit a pull request.

**Roadmap:**

* [ ] Performance optimization (the biggest task).
* [ ] Add more loss functions (e.g., `CrossEntropy`).
* [ ] Add more metrics (e.g., `Accuracy`, `Precision`).
* [ ] Improve the `SNetwork.fit()` method with `batch_size`, validation splits, and callbacks.
* [ ] Long-term: Explore Dart FFI for GPU acceleration via C++ libraries.

**How to Contribute:**

* If you find a bug or have a feature request, please file an issue.
* Feel free to fork the repository and submit a pull request.