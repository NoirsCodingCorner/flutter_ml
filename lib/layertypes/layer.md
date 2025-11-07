## `Layer` Module Class

The `Layer` class is the abstract base class for all neural network layers. It is the fundamental, callable building block of a neural network.

A `Layer` is designed to be chained together in a model, where the output of one layer becomes the input to the next. It encapsulates two key things:

* **State:** Its trainable `parameters` (e.g., weights and biases).
* **Transformation:** The logic in its `forward` pass that transforms input tensors to output tensors.

### Layer Lifecycle

The typical lifecycle of a layer is:

1.  **Instantiation:** A layer is created (e.g., `DenseLayer(64)`). Its weights and biases are **not** created yet.
2.  **Build:** The first time the layer is called (e.g., `layer.call(input)`), the `build` method runs automatically. This method uses the input's shape to initialize the `parameters` with the correct dimensions. This is known as deferred initialization.
3.  **Forward Pass:** On every call, the `forward` method is executed to perform the layer's core mathematical operations.

### Example Usage

```dart
// 1. Define a layer (weights are not created yet).
Layer dense = DenseLayer(32);

// 2. Create an input tensor.
Tensor<Vector> input = Tensor<Vector>([1.0, 2.0, 3.0]);

// 3. Call the layer. 
// This will (1) build the layer, then (2) run the forward pass.
Tensor<Vector> output = dense.call(input) as Tensor<Vector>;
```

### Properties

* `List<Tensor> get parameters`
    * (Abstract) A list of all trainable tensors (weights and biases) in the layer. This getter is used by the `Optimizer` to know which tensors to update during training.
* `String get name`
    * (Abstract) A user-friendly name for the layer (e.g., 'dense', 'lstm').
* `bool _built`
    * (Private) A boolean flag to check if the `build` method has been run at least once.

### Methods

These are the main methods for using and defining a layer.

* `Tensor<dynamic> call(Tensor<dynamic> input)`
    * The public, callable interface for the layer. This wrapper method automatically handles the `build` step on the first run before executing the main `forward` pass logic. **You should always use `.call()` to run a layer.**
* `Tensor<dynamic> forward(Tensor<dynamic> input)`
    * (Abstract) The core logic of the layer's transformation. Subclasses **must** implement this method to define how they process an input tensor and return an output tensor.
* `void build(Tensor<dynamic> input)`
    * This method initializes the layer's parameters based on the shape of the first input. Subclasses should override this method to create their weights and biases. It is called automatically by `call` and should not be called directly.

## The standart available Layer types

* **`DenseLayer`**: A standard fully-connected layer for 1D `Vector` inputs.
* **`DenseLayerMatrix`**: A fully-connected layer that processes a 2D `Matrix`, applying the dense operation to each row.
* **`RNN`**: A simple Recurrent Neural Network (RNN) layer for sequence data.
* **`LSTMLayer`**: A Long Short-Term Memory (LSTM) recurrent layer for sequence data (input is a `Matrix`), designed to learn long-term dependencies.
* **`DualLSTMLayer`**: A custom hierarchical LSTM with two internal tiers (a fast "lower" tier and a slow "higher" tier) to model dependencies across different timescales.
* **`MultiTierLSTMLayer`**: A generalized, hierarchical LSTM with an arbitrary number of configured timescales.
* **`GeneralizedChainedScaleLayer`**: A self-contained, multi-scale recurrent layer for processing high-frequency data by creating and chaining lower-frequency summaries.
* **`Conv2DLayer`**: A 2D convolutional layer that applies filters to a 2D `Matrix` input, producing a 3D `Tensor3D` (multi-channel) output.
* **`ConvLSTMLayer`**: A Convolutional LSTM layer for spatiotemporal data (like video). It uses convolutions inside its gates instead of matrix multiplication.
* **`MaxPooling1DLayer`**: Downsamples a 1D `Vector` sequence by taking the maximum value over a window.
* **`MaxPooling2DLayer`**: Downsamples a 2D `Matrix` feature map by taking the maximum value over a window.
* **`AveragePooling2DLayer`**: Downsamples a 2D `Matrix` feature map by taking the average value over a window.
* **`GlobalAveragePoolingLayer`**: Converts a 2D `Matrix` (like `[seq_len, features]`) into a 1D `Vector` (`[features]`) by averaging all rows.
* **`GlobalAveragePooling1D`**: A Global Average Pooling layer for 1D data (`[sequence_length, features]` to `[features]`).
* **`BatchNorm1D`**: Stabilizes training by normalizing a 1D `Vector` input.
* **`BatchNorm2D`**: Stabilizes training by normalizing a 3D `Tensor3D` input (e.g., after a `Conv2DLayer`), applying normalization per-channel.
* **`LayerNormalization`**: Normalizes inputs across the feature dimension for a 2D `Matrix` (e.g., a batch or sequence).
* **`LayerNormalizationVector`**: Normalizes inputs across the feature dimension for a 1D `Vector`.
* **`DropoutLayer`**: A regularization layer for `Vector` inputs. Randomly sets inputs to 0 during training to prevent overfitting.
* **`DropoutLayerMatrix`**: A regularization layer for `Matrix` inputs, applying dropout to each element.
* **`FlattenLayer`**: A utility layer with no parameters that reshapes a multi-dimensional input (like a `Matrix`) into a 1D `Vector`.
* **`ReshapeVectorToMatrixLayer`**: A utility layer to reshape a `Vector` to a 1x1 `Matrix`.
* **`ReLULayer`**: An activation layer that applies ReLU to a `Vector`.
* **`ReLULayerMatrix`**: An activation layer that applies ReLU to a `Matrix`.
* **`EmbeddingLayer`**: Converts a 1D `Vector` of integer indices into a 2D `Matrix` of dense vectors.
* **`EmbeddingLayerMatrix`**: Converts a 2D `Matrix` (batch) of integer indices into a 3D `Tensor3D` of dense vectors.
* **`PositionalEncoding`**: Injects non-trainable sinusoidal position information into sequence embeddings.
* **`SingleHeadAttention`**: Implements a single head of the self-attention mechanism.
* **`MultiHeadAttention`**: Implements the Multi-Head Self-Attention mechanism by running multiple `SingleHeadAttention` heads in parallel.
* **`TransformerEncoderBlock`**: A single Transformer Encoder Block, combining Multi-Head Attention and a Feed-Forward Network.