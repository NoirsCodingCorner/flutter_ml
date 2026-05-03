
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/relu.dart';
import '../layertypes/layer.dart';
import '../networks/SNetwork.dart';

class TransformerEncoderBlock extends Layer<Matrix, Matrix> {
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
  List<Tensor> get parameters {
    List<Tensor> params = [];
    params.addAll(mha.parameters);
    params.addAll(layerNorm1.parameters);
    params.addAll(ffn.parameters);
    params.addAll(layerNorm2.parameters);
    return params;
  }

  @override
  void build(Tensor<Matrix> input) {
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
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    Tensor<Matrix> attentionOutput = mha.call(input);

    Tensor<Matrix> added1 = addMatrix(input, attentionOutput);
    Tensor<Matrix> addAndNorm1 = layerNorm1.call(added1);

    // Assuming SNetwork returns dynamic or un-typed Tensor, casting here.
    // If SNetwork is updated to Layer<Matrix, Matrix>, the cast can be removed.
    Tensor<Matrix> ffnOutput = ffn.call(addAndNorm1) as Tensor<Matrix>;

    Tensor<Matrix> added2 = addMatrix(addAndNorm1, ffnOutput);
    Tensor<Matrix> addAndNorm2 = layerNorm2.call(added2);

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

