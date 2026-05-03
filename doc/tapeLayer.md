# TapeLayer Overview

The `TapeLayer` is the core architectural component of the neural network builder in. It is an abstract class designed to encapsulate trainable parameters (like weights and biases) and manage the construction of the mathematical graph via the `CommandBuffer` (the "tape").

## How `TapeLayer` Works

A `TapeLayer` manages three primary responsibilities:

1. **State & Parameter Management:** It holds references to `GPUTensor` instances that represent the layer's learnable parameters. The `parameters` getter exposes these to the optimizer so they can be updated during training.
2. **Dynamic Building (`build`):** The `build(GPUTensor input)` method is called exactly once when the layer is first executed. It uses the shape of the incoming tensor to calculate and allocate the precise VRAM dimensions needed for its weights and biases, often initializing them using specific strategies (like Xavier initialization).
3. **Graph Construction (`forward`):** Instead of executing math directly, the `forward` method maps the layer's logic into GPU OpCodes. It takes the `input` tensor and a shared `CommandBuffer` tape, dispatching the necessary GPU math functions (e.g., `matMulGPU`, `reluMatrixGPU`) onto the tape. It also populates an `intermediates` list with transient tensors, ensuring that temporary VRAM allocations can be safely garbage collected after the tape executes.

## Available TapeLayers

Here is the complete list of available implementations of the `TapeLayer` in the provided code:

* `AveragePooling2DGPU`
* `BatchNorm1DGPU`
* `BatchNorm2DGPU`
* `Conv2DTapeLayer`
* `ConvLSTMTapeLayer`
* `DenseLayer`
* `DenseReluLayer` (A fused layer for better performance)
* `DropoutMatrixTapeLayer`
* `DropoutTapeLayer`
* `DualLSTMTapeLayer`
* `EmbeddingMatrixTapeLayer`
* `EmbeddingTapeLayer`
* `FlattenLayer`
* `GeluLayerMatrixTapeLayer`
* `GeluLayerTapeLayer`
* `GlobalAveragePooling1DTapeLayer`
* `GlobalAveragePoolingGPU`
* `LSTMTapeLayer`
* `LayerNormalizationTapeLayer`
* `MaxPooling1DTapeLayer`
* `MaxPooling2DTapeLayer`
* `MultiHeadAttentionTapeLayer`
* `PositionalEncodingTapeLayer`
* `RNNTapeLayer`
* `ReLULayerMatrixTapeLayer`
* `ReLULayerTapeLayer`
* `SingleHeadAttentionTapeLayer`
* `TransformerEncoderBlockTapeLayer`