import 'dart:math';

import '../activationFunctions/softmax.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// Implements a single head of the self-attention mechanism.
///
/// This layer learns the contextual relationships between tokens in a sequence
/// by calculating Query, Key, and Value projections and producing a weighted
/// average of the values based on the query-key similarity.
/// Implements a single head of the self-attention mechanism.
///
/// This layer is the core of the Transformer architecture. It learns contextual
/// relationships between tokens in a sequence. It does this by creating Query (Q),
/// Key (K), and Value (V) projections for each input token and then producing a
/// weighted average of the Values based on the Query-Key similarity.
///
/// The attention formula is: $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
///
/// - **Input:** A `Tensor<Matrix>` of shape `[sequence_length, dModel]`.
/// - **Output:** A `Tensor<Matrix>` of shape `[sequence_length, dV]`.
/// Implements a single head of the self-attention mechanism.
/// Implements a single head of the self-attention mechanism.
class SingleHeadAttention extends Layer {
  @override
  String name = 'single_head_attention';
  int dModel;
  int dK;
  int dV;

  late Tensor<Matrix> Wq;
  late Tensor<Matrix> Wk;
  late Tensor<Matrix> Wv;

  /// Stores the attention weights from the most recent forward pass.
  /// Useful for debugging and visualization.
  late Tensor<Matrix> lastAttentionWeights;

  SingleHeadAttention(this.dModel, {int? dK, int? dV})
      : dK = dK ?? dModel,
        dV = dV ?? dModel;

  @override
  List<Tensor> get parameters => [Wq, Wk, Wv];

  @override
  void build(Tensor<dynamic> input) {
    Random random = Random();
    Tensor<Matrix> initWeights(int rows, int cols) {
      double stddev = sqrt(1.0 / rows);
      Matrix values = [];
      for (int i = 0; i < rows; i++) {
        Vector row = [];
        for (int j = 0; j < cols; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }
    Wq = initWeights(dModel, dK);
    Wk = initWeights(dModel, dK);
    Wv = initWeights(dModel, dV);
    super.build(input);
  }

  /// The forward pass now correctly returns only a Tensor<Matrix>.
  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;

    Tensor<Matrix> Q = matMul(inputMatrix, Wq);
    Tensor<Matrix> K = matMul(inputMatrix, Wk);
    Tensor<Matrix> V = matMul(inputMatrix, Wv);

    Tensor<Matrix> Kt = transpose(K);
    Tensor<Matrix> scores = matMul(Q, Kt);

    Tensor<Matrix> scaledScores = scaleMatrix(scores, 1 / sqrt(dK));
    Tensor<Matrix> attentionWeights = softmaxMatrix(scaledScores);

    // Store the weights in the public property for inspection.
    lastAttentionWeights = attentionWeights;

    Tensor<Matrix> out = matMul(attentionWeights, V);

    return out;
  }
}


/*void main() {
  int dModel = 4;
  int dK = 2;

  SingleHeadAttention attentionLayer = SingleHeadAttention(dModel, dK: dK, dV: dK);

  Tensor<Matrix> inputSequence = Tensor<Matrix>([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
  ]);

  // The 'call' method builds the layer and runs the forward pass.
  Tensor<Matrix> outputSequence = attentionLayer.call(inputSequence) as Tensor<Matrix>;

  // After the forward pass, access the stored weights.
  Tensor<Matrix> attentionWeights = attentionLayer.lastAttentionWeights;

  print('--- Single Head Attention Test ---');
  print('\nInput Sequence (shape: ${inputSequence.value.length}x${inputSequence.value[0].length}):');
  print(inputSequence.value);

  print('\nOutput Sequence (shape: ${outputSequence.value.length}x${outputSequence.value[0].length}):');
  outputSequence.value.forEach((row) => print(row.map((e) => e.toStringAsFixed(4)).toList()));

  print('\nAttention Weights Matrix (shape: ${attentionWeights.value.length}x${attentionWeights.value[0].length}):');
  attentionWeights.value.forEach((row) => print(row.map((e) => e.toStringAsFixed(4)).toList()));
  outputSequence.printGraph();
}*/