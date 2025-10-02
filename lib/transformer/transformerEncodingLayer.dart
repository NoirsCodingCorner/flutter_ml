import 'package:flutter_ml/transformer/positionalEncodingLayer.dart';

import '../activationFunctions/relu.dart';
import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/layer.dart';
import '../nets/snet.dart';
import 'embeddingLayer.dart';
import 'layerNormalization.dart';
import 'multiHeadAttentionLayer.dart';

/// Implements a single Transformer Encoder Block.
///
/// This is the main repeating component of the Transformer's encoder. It consists
/// of two primary sub-layers: a Multi-Head Self-Attention mechanism and a
/// position-wise Feed-Forward Network. Each sub-layer is followed by a
/// residual connection and a layer normalization step.
///
/// By stacking these blocks, the model can build an increasingly deep and
/// context-aware representation of the input sequence.
class TransformerEncoderBlock extends Layer {
  @override
  String name = 'transformer_encoder_block';
  int dModel;
  int numHeads;
  int dff;

  late MultiHeadAttention mha;
  late LayerNormalization layerNorm1;
  late SNetwork ffn;
  late LayerNormalization layerNorm2;

  TransformerEncoderBlock(this.dModel, this.numHeads, this.dff);

  @override
  List<Tensor> get parameters => [
    ...mha.parameters,
    ...layerNorm1.parameters,
    ...ffn.parameters,
    ...layerNorm2.parameters,
  ];

  @override
  void build(Tensor<dynamic> input) {
    mha = MultiHeadAttention(dModel, numHeads);
    layerNorm1 = LayerNormalization();

    ffn = SNetwork([
      DenseLayerMatrix(dff, activation: ReLUMatrix()),
      DenseLayerMatrix(dModel),
    ]);

    layerNorm2 = LayerNormalization();
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;

    Tensor<Matrix> attentionOutput = mha.call(inputMatrix) as Tensor<Matrix>;
    Tensor<Matrix> addAndNorm1 =
    layerNorm1.call(addMatrix(inputMatrix, attentionOutput)) as Tensor<Matrix>;

    Tensor<Matrix> ffnOutput = ffn.call(addAndNorm1) as Tensor<Matrix>;
    Tensor<Matrix> addAndNorm2 =
    layerNorm2.call(addMatrix(addAndNorm1, ffnOutput)) as Tensor<Matrix>;

    return addAndNorm2;
  }
}

void main() {
  int vocabSize = 1000;
  int dModel = 32;
  int numHeads = 4;
  int dff = 64;
  int numEncoderBlocks = 2; // A 2-layer Transformer

  // 1. Assemble the full model architecture
  List<Layer> modelLayers = [
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(100, dModel),
  ];
  for (int i = 0; i < numEncoderBlocks; i++) {
    modelLayers.add(TransformerEncoderBlock(dModel, numHeads, dff));
  }
  // For classification, you would typically add a final Dense layer.

  SNetwork transformer = SNetwork(modelLayers);

  // 2. Build the model with a dummy input
  Tensor<Vector> dummySentence = Tensor<Vector>([10, 20, 30]);
  transformer.predict(dummySentence);

  print('--- Full Transformer Encoder Model ---');
  print('Model built successfully with ${numEncoderBlocks} encoder blocks.');

  // 3. Print the graph to see the complete structure
  transformer.predict(dummySentence).printGraph(); // Use maxDepth to keep it readable
}