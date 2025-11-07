
## `Transformer Model` Showcase

This simple autograd engine is fully capable of building and running modern, complex architectures like the Transformer. 
This section provides all the necessary building blocks to create your own Transformer Encoder for tasks like text classification, sequence-to-sequence, and more.

**--- Be advised that this engine is NOT YET optimised for speed or scalability due to me being just one person ---** 

All the necessary layers are provided as modular components, allowing you to stack them to build a full model.

### Core Transformer Building Blocks

These are the fundamental layers available for constructing a Transformer.

* **`EmbeddingLayer` / `EmbeddingLayerMatrix`**

    * Converts a 1D vector (or 2D batch) of integer word indices (tokens) into a 2D matrix (or 3D batch) of dense "embedding" vectors. This is the first step in any NLP model.

* **`PositionalEncoding`**

    * A non-trainable layer that adds sinusoidal information about the relative or absolute position of tokens in a sequence. Since the Transformer has no "recurrence" (like an RNN), this is essential for it to understand word order.

* **`MultiHeadAttention`**

    * The core of the Transformer. This layer runs multiple "attention heads" in parallel, allowing the model to jointly attend to information from different representation "subspaces" at different positions.

* **`LayerNormalization`**

    * A critical component for stabilizing the deep network. Unlike `BatchNorm`, it normalizes the activations *across the features* for each sequence, making it independent of batch size.

* **`TransformerEncoderBlock`**

    * This is the main repeating unit of the Transformer. It's a "meta-layer" that combines:
        1.  A `MultiHeadAttention` layer
        2.  A residual ("add") connection
        3.  A `LayerNormalization` layer
        4.  A position-wise Feed-Forward Network (FFN)
        5.  A second residual connection & `LayerNormalization`

* **`GlobalAveragePooling1D`**

    * A utility layer often used at the end of an encoder. It takes the final output matrix (e.g., `[seq_length, d_model]`) and computes the average of all tokens, producing a single vector (`[d_model]`) that represents the entire sequence.

### How They Work Together: An Encoder

You can build a complete Transformer Encoder by stacking these layers in an `SNetwork`. The data flows through the model as follows:

1.  A 1D `Tensor<Vector>` of token indices (e.g., `[1, 2, 3]`) is fed into the model.
2.  **`EmbeddingLayer`** turns this into a 2D matrix (`[seq_len, d_model]`).
3.  **`PositionalEncoding`** adds the word order information to this matrix.
4.  The matrix is passed through one or more **`TransformerEncoderBlock`** layers. Each block refines the "contextual" meaning of each token by looking at all other tokens.
5.  The final output matrix from the last block (`[seq_len, d_model]`) is "pooled" by **`GlobalAveragePooling1D`** into a single `Tensor<Vector>`.
6.  This single vector, which now represents the "meaning" of the entire sentence, is fed into a final **`DenseLayer`** for classification (e.g., to get a 0.0 to 1.0 sentiment score).

### Complete Example: Sentiment Analysis

The `finalModel.dart` file demonstrates how to assemble these components into a complete, trainable model for sentiment analysis.

Here is how the `SNetwork` is defined in that file, showing the simple, sequential stacking of all the layers:

```dart
/*
  This code snippet from 'finalModel.dart' shows how all the
  building blocks are assembled into a single SNetwork.
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