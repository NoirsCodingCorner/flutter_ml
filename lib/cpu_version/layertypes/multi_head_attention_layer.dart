import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_aliases.dart';
import '../layertypes/layer.dart';

/// Applies Multi-Head Attention over an input [Matrix] tensor.
/// Projects the input sequence into multiple subspaces by running [numHeads] parallel [SingleHeadAttention] modules,
/// concatenates the head outputs along the column dimension, and linearly projects the result back to [dModel] using [wo].
class MultiHeadAttention extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'multi_head_attention';

  /// Total dimensionality of the input and output feature representations.
  int dModel;

  /// Number of parallel attention heads.
  int numHeads;

  /// List of independent [SingleHeadAttention] heads computing attention in parallel.
  late List<SingleHeadAttention> attentionHeads;

  /// Final linear output projection weight matrix tensor of shape `[dModel, dModel]`.
  late Tensor<Matrix> wo;

  /// Creates a [MultiHeadAttention] layer with [dModel] feature size split evenly across [numHeads].
  /// Throws an exception if [dModel] is not evenly divisible by [numHeads].
  MultiHeadAttention(this.dModel, this.numHeads) {
    if (dModel % numHeads != 0) {
      throw Exception('dModel must be divisible by numHeads.');
    }
  }

  /// Returns all trainable parameters across all individual attention heads and the output projection matrix [wo].
  @override
  List<Tensor> get parameters {
    List<Tensor> params = [];
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      params.addAll(attentionHeads[i].parameters);
    }
    params.add(wo);
    return params;
  }

  /// Instantiates each [SingleHeadAttention] head with `dHead = dModel ~/ numHeads`
  /// and initializes the output projection matrix [wo] using uniform Xavier scaling.
  @override
  void build(Tensor<Matrix> input) {
    int dHead = dModel ~/ numHeads;
    attentionHeads = [];
    for (int i = 0; i < numHeads; i = i + 1) {
      attentionHeads.add(SingleHeadAttention(dModel, dK: dHead, dV: dHead));
    }

    Random random = Random();
    List<double> woValues = [];
    double limit = sqrt(1.0 / dModel);
    int totalElements = dModel * dModel;

    for (int i = 0; i < totalElements; i = i + 1) {
      woValues.add((random.nextDouble() * 2 - 1) * limit);
    }

    wo = Tensor<Matrix>(woValues);
    wo.shape = [dModel, dModel];

    super.build(input);
  }

  /// Executes multi-head self-attention on the CPU by computing all head outputs,
  /// concatenating them along columns, and projecting through [wo].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    List<Tensor<Matrix>> headOutputs = [];
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      Tensor<Matrix> headOutput = attentionHeads[i].call(input);
      headOutputs.add(headOutput);
    }

    Tensor<Matrix> concatenatedOutput = concatenateMatricesByColumn(headOutputs);
    Tensor<Matrix> finalOutput = matMul(concatenatedOutput, wo);

    return finalOutput;
  }

  /// Returns the output projection matrix [wo] and all attention head weight dictionaries as a map.
  @override
  Map<String, dynamic> getWeights() {
    List<Map<String, dynamic>> headWeights = [];
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      headWeights.add(attentionHeads[i].getWeights());
    }

    return {
      'Wo': wo.value,
      'heads': headWeights,
    };
  }

  /// Sets the output projection matrix [wo] and delegates weight setting to each head from the weights map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> woDynamic = weightsMap['Wo'] as List<dynamic>;

    for (int i = 0; i < dModel; i = i + 1) {
      List<dynamic> row = woDynamic[i] as List<dynamic>;
      int rowOffset = i * dModel;
      for (int j = 0; j < dModel; j = j + 1) {
        wo.data[rowOffset + j] = row[j] as double;
      }
    }

    List<dynamic> headWeights = weightsMap['heads'] as List<dynamic>;
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      attentionHeads[i].setWeights(headWeights[i] as Map<String, dynamic>);
    }
  }
}
