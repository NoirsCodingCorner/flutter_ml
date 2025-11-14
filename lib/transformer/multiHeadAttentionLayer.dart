import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';

class MultiHeadAttention extends Layer {
  @override
  String name = 'multi_head_attention';
  int dModel;
  int numHeads;

  late List<SingleHeadAttention> attentionHeads;
  late Tensor<Matrix> Wo;

  MultiHeadAttention(this.dModel, this.numHeads) {
    if (dModel % numHeads != 0) {
      throw Exception('dModel must be divisible by numHeads.');
    }
  }

  @override
  List<Tensor> get parameters {
    List<Tensor> params = [];
    for (SingleHeadAttention head in attentionHeads) {
      params.addAll(head.parameters);
    }
    params.add(Wo);
    return params;
  }

  @override
  void build(Tensor<dynamic> input) {
    int dHead = dModel ~/ numHeads;
    attentionHeads = [];
    for (int i = 0; i < numHeads; i++) {
      attentionHeads.add(SingleHeadAttention(dModel, dK: dHead, dV: dHead));
    }

    Random random = Random();
    Matrix wo_values = [];
    for(int i=0; i<dModel; i++){
      Vector row = [];
      for(int j=0; j<dModel; j++){
        row.add((random.nextDouble() * 2 - 1) * sqrt(1.0 / dModel));
      }
      wo_values.add(row);
    }
    Wo = Tensor<Matrix>(wo_values);

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    List<Tensor<Matrix>> headOutputs = [];
    for (SingleHeadAttention head in attentionHeads) {
      Tensor<Matrix> headOutput = head.call(input) as Tensor<Matrix>;
      headOutputs.add(headOutput);
    }

    Tensor<Matrix> concatenatedOutput = concatenateMatricesByColumn(headOutputs);
    Tensor<Matrix> finalOutput = matMul(concatenatedOutput, Wo);

    return finalOutput;
  }

  @override
  Map<String, dynamic> getWeights() {
    List<Map<String, dynamic>> headWeights = [];
    for (SingleHeadAttention head in attentionHeads) {
      headWeights.add(head.getWeights());
    }

    return {
      'Wo': Wo.value,
      'heads': headWeights,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> woDynamic = weightsMap['Wo'] as List<dynamic>;
    Matrix newWo = woDynamic.map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();

    for (int i = 0; i < Wo.value.length; i++) {
      for (int j = 0; j < Wo.value[0].length; j++) {
        Wo.value[i][j] = newWo[i][j];
      }
    }

    List<dynamic> headWeights = weightsMap['heads'] as List<dynamic>;
    for (int i = 0; i < attentionHeads.length; i++) {
      attentionHeads[i].setWeights(headWeights[i] as Map<String, dynamic>);
    }
  }
}