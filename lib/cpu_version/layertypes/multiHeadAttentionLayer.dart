import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../layertypes/layer.dart';

class MultiHeadAttention extends Layer<Matrix, Matrix> {
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
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      params.addAll(attentionHeads[i].parameters);
    }
    params.add(Wo);
    return params;
  }

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

    Wo = Tensor<Matrix>(woValues);
    Wo.shape = [dModel, dModel];

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    List<Tensor<Matrix>> headOutputs = [];
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      Tensor<Matrix> headOutput = attentionHeads[i].call(input);
      headOutputs.add(headOutput);
    }

    Tensor<Matrix> concatenatedOutput = concatenateMatricesByColumn(headOutputs);
    Tensor<Matrix> finalOutput = matMul(concatenatedOutput, Wo);

    return finalOutput;
  }

  @override
  Map<String, dynamic> getWeights() {
    List<Map<String, dynamic>> headWeights = [];
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      headWeights.add(attentionHeads[i].getWeights());
    }

    return {
      'Wo': Wo.value,
      'heads': headWeights,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> woDynamic = weightsMap['Wo'] as List<dynamic>;

    for (int i = 0; i < dModel; i = i + 1) {
      List<dynamic> row = woDynamic[i] as List<dynamic>;
      int rowOffset = i * dModel;
      for (int j = 0; j < dModel; j = j + 1) {
        Wo.data[rowOffset + j] = row[j] as double;
      }
    }

    List<dynamic> headWeights = weightsMap['heads'] as List<dynamic>;
    for (int i = 0; i < attentionHeads.length; i = i + 1) {
      attentionHeads[i].setWeights(headWeights[i] as Map<String, dynamic>);
    }
  }
}