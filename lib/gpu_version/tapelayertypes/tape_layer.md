# TapeLayer Overview

The `TapeLayer` is the foundational abstraction for constructing complex neural networks and mathematical models within this library. 
Designed as a strongly-typed abstract class, it strictly enforces an `InputType` and `OutputType` to guarantee type safety across the entire network architecture. 
Its primary role is to encapsulate trainable parameters, manage `GPUTensor` lifecycles, and coordinate the deferred construction of the mathematical execution graph.

## Core Architecture & Lifecycle

A `TapeLayer` operates through several distinct phases and responsibilities to ensure high-performance GPU execution and memory safety:

* **State & Parameter Management:** The layer encapsulates all learnable parameters (weights, biases) as persistent `GPUTensor` instances. 
The `parameters` getter exposes these directly to the optimizer, allowing for seamless backpropagation and gradient updates during training.
* **Dynamic VRAM Allocation (`build`):** Executed exactly once during the layer's initial run, the `build(GPUTensor<InputType> input)` method uses the incoming tensor's shape to calculate and allocate the precise VRAM required. 
This phase also handles the structural initialization of weights (such as Xavier or scaled random initialization).
* **Deferred Graph Construction (`forward`):** Rather than executing computations eagerly, the `forward` method maps the layer's logic into GPU OpCodes. 
It takes the `input` tensor and a shared `CommandBuffer` (the "tape"), subsequently dispatching the necessary GPU math functions (e.g., `matMulGPU`, `reluMatrixGPU`) onto the tape for deferred batch execution.
* **Transient Memory Management:** During the graph construction phase, the layer populates an `intermediates` list with transient tensors. 
This ensures that temporary VRAM allocations generated during complex, multi-step operations can be safely tracked and garbage-collected immediately after the tape executes.
* **Persistent Caching & Static Unrolling:** To maximize runtime performance, all native `TapeLayer` implementations support **static tape unrolling**. 
This guarantees that no new VRAM is allocated during the standard inference or training loop. 
Layers with recurrent or complex internal states actively manage their own cached memory tensors, reusing them across steps to prevent runtime reallocation and memory fragmentation.

## Available TapeLayers

The library provides a comprehensive suite of pre-built `TapeLayer` implementations for various architectural needs:

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