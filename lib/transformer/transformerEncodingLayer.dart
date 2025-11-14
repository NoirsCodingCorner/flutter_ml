import 'package:flutter_ml/transformer/positionalEncodingLayer.dart';

import '../activationFunctions/relu.dart';
import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';
import '../nets/snet.dart';
import 'embeddingLayer.dart';
import 'layerNormalization.dart';
import 'multiHeadAttentionLayer.dart';

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

  @override
  Map<String, dynamic> getWeights() {
    return {
      'mha': mha.getWeights(),
      'layerNorm1': layerNorm1.getWeights(),
      'ffn': ffn.getWeights(),
      'layerNorm2': layerNorm2.getWeights(),
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    mha.setWeights(weightsMap['mha'] as Map<String, dynamic>);
    layerNorm1.setWeights(weightsMap['layerNorm1'] as Map<String, dynamic>);
    ffn.setWeights(weightsMap['ffn'] as Map<String, dynamic>);
    layerNorm2.setWeights(weightsMap['layerNorm2'] as Map<String, dynamic>);
  }
}