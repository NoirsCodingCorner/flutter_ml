import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import '../activationFunctions/relu.dart';
import '../layertypes/layer.dart';
import '../networks/s_network.dart';

/// Applies a standard Transformer Encoder Block over an input sequence [Matrix] tensor.
/// Combines a multi-head self-attention module ([mha]) with a position-wise feed-forward network ([ffn]).
/// Each sub-layer incorporates residual connections ([addMatrix]) followed by layer normalization ([LayerNormalization]).
class TransformerEncoderBlock extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'transformer_encoder_block';

  /// Total dimensionality of the input, intermediate, and output token representations.
  int dModel;

  /// Number of parallel attention heads in the multi-head attention sub-layer.
  int numHeads;

  /// Inner dimensional size of the position-wise feed-forward network.
  int dff;

  /// Multi-head self-attention sub-layer module.
  late MultiHeadAttention mha;

  /// First layer normalization applied after the residual addition of the attention sub-layer.
  late LayerNormalization layerNorm1;

  /// Position-wise feed-forward neural network containing two dense layers with ReLU activation.
  late SNetwork ffn;

  /// Second layer normalization applied after the residual addition of the feed-forward network.
  late LayerNormalization layerNorm2;

  /// Creates a [TransformerEncoderBlock] with model dimensionality [dModel], [numHeads] attention heads, and [dff] feed-forward units.
  TransformerEncoderBlock(this.dModel, this.numHeads, this.dff);

  /// Returns all trainable parameters aggregated across [mha], [layerNorm1], [ffn], and [layerNorm2].
  @override
  List<Tensor> get parameters {
    List<Tensor> params = [];
    params.addAll(mha.parameters);
    params.addAll(layerNorm1.parameters);
    params.addAll(ffn.parameters);
    params.addAll(layerNorm2.parameters);
    return params;
  }

  /// Instantiates sub-layers ([mha], [layerNorm1], [ffn], and [layerNorm2]) and registers the block as built.
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

  /// Executes the transformer encoder forward pass on the CPU:
  /// Applies multi-head attention, residual addition, first normalization, feed-forward network, residual addition, and second normalization.
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

  /// Returns the serialized weights of all constituent sub-layers as a nested map.
  @override
  Map<String, dynamic> getWeights() {
    return {
      'mha': mha.getWeights(),
      'layerNorm1': layerNorm1.getWeights(),
      'ffn': ffn.getWeights(),
      'layerNorm2': layerNorm2.getWeights(),
    };
  }

  /// Delegates setting weights to [mha], [layerNorm1], [ffn], and [layerNorm2] from a provided nested map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    mha.setWeights(weightsMap['mha'] as Map<String, dynamic>);
    layerNorm1.setWeights(weightsMap['layerNorm1'] as Map<String, dynamic>);
    ffn.setWeights(weightsMap['ffn'] as Map<String, dynamic>);
    layerNorm2.setWeights(weightsMap['layerNorm2'] as Map<String, dynamic>);
  }
}