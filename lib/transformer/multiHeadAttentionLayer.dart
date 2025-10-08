import 'dart:math';

import 'package:flutter_ml/layertypes/singleHeadAttentionLayer.dart';

import '../autogradEngine/tensor.dart';
import '../layertypes/layer.dart';

/// Implements the Multi-Head Self-Attention mechanism.
///
/// This layer runs multiple `SingleHeadAttention` heads in parallel, allowing the
/// model to jointly attend to information from different representation subspaces.
/// The outputs of the heads are concatenated and passed through a final linear
/// projection.
///
/// This is the core component of the Transformer architecture.
/// Implements the Multi-Head Self-Attention mechanism.
///
/// This layer runs multiple `SingleHeadAttention` heads in parallel, allowing the
/// model to jointly attend to information from different representation subspaces.
/// The outputs of the heads are concatenated and passed through a final linear
/// projection.
///
/// This is the core component of the Transformer architecture.
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
      // Call the head, which correctly returns its output Tensor.
      Tensor<Matrix> headOutput = head.call(input) as Tensor<Matrix>;
      headOutputs.add(headOutput);
    }

    Tensor<Matrix> concatenatedOutput = concatenateMatricesByColumn(headOutputs);
    Tensor<Matrix> finalOutput = matMul(concatenatedOutput, Wo);

    return finalOutput;
  }
}


/*void main() {
  int sequenceLength = 3;
  int dModel = 4;
  int numHeads = 4;

  MultiHeadAttention attentionLayer = MultiHeadAttention(dModel, numHeads);

  Tensor<Matrix> inputSequence = Tensor<Matrix>([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
  ]);

  Tensor<Matrix> outputSequence = attentionLayer.call(inputSequence) as Tensor<Matrix>;

  print('--- Multi-Head Attention Test ---');
  print('\nInput Sequence (shape: ${inputSequence.value.length}x${inputSequence.value[0].length}):');
  print(inputSequence.value);

  print('\nFinal Output Sequence (shape: ${outputSequence.value.length}x${outputSequence.value[0].length}):');
  for (Vector row in outputSequence.value) {
    print(row.map((double e) => e.toStringAsFixed(4)).toList());
  }

  print('\n--- Individual Head Attention Weights ---');
  for (int i = 0; i < attentionLayer.attentionHeads.length; i++) {
    Tensor<Matrix> attentionWeights = attentionLayer.attentionHeads[i].lastAttentionWeights;
    print('\nAttention Weights for Head $i (shape: ${attentionWeights.value.length}x${attentionWeights.value[0].length}):');
    for (Vector row in attentionWeights.value) {
      print(row.map((double e) => e.toStringAsFixed(4)).toList());
    }
  }
  outputSequence.printGraph();
}*/

