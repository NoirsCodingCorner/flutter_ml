import 'dart:math';


import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

/// Applies single-head scaled dot-product attention over an input sequence [Matrix] tensor.
/// Projects input token embeddings into Query ([Wq]), Key ([Wk]), and Value ([Wv]) representations,
/// computes attention scores scaled by `1.0 / sqrt(dK)`, applies softmax normalization to extract attention weights,
/// and computes the context matrix by weighting [Wv].
class SingleHeadAttention extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'single_head_attention';

  /// Dimensionality of the input and output token representation.
  int dModel;

  /// Dimensionality of the query and key projections.
  int dK;

  /// Dimensionality of the value projection.
  int dV;

  /// Query projection weight matrix tensor of shape `[dModel, dK]`.
  late Tensor<Matrix> Wq;

  /// Key projection weight matrix tensor of shape `[dModel, dK]`.
  late Tensor<Matrix> Wk;

  /// Value projection weight matrix tensor of shape `[dModel, dV]`.
  late Tensor<Matrix> Wv;

  /// Caches the attention weights matrix from the most recent forward pass for inspection.
  late Tensor<Matrix> lastAttentionWeights;

  /// Creates a [SingleHeadAttention] layer with [dModel] feature size and optional projection dimensions [dK] and [dV].
  SingleHeadAttention(this.dModel, {int? dK, int? dV})
      : dK = dK ?? dModel,
        dV = dV ?? dModel;

  /// Returns all trainable parameters: [Wq], [Wk], and [Wv].
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(Wq);
    params.add(Wk);
    params.add(Wv);
    return params;
  }

  /// Allocates and initializes projection matrices [Wq], [Wk], and [Wv] using uniform fan-in scaling.
  @override
  void build(Tensor<Matrix> input) {
    Random random = Random();

    Tensor<Matrix> initWeights(int rows, int cols) {
      double stddev = sqrt(1.0 / rows);
      Matrix values = [];
      for (int i = 0; i < rows; i = i + 1) {
        Vector row = [];
        for (int j = 0; j < cols; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
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

  /// Executes scaled dot-product self-attention on the CPU, caches [lastAttentionWeights], and returns the projected [Matrix].
  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    Tensor<Matrix> Q = matMul(input, Wq);
    Tensor<Matrix> K = matMul(input, Wk);
    Tensor<Matrix> V = matMul(input, Wv);

    Tensor<Matrix> Kt = transpose(K);
    Tensor<Matrix> scores = matMul(Q, Kt);

    Tensor<Matrix> scaledScores = scaleMatrix(scores, 1.0 / sqrt(dK));
    Tensor<Matrix> attentionWeights = softmaxMatrix(scaledScores);

    lastAttentionWeights = attentionWeights;

    Tensor<Matrix> out = matMul(attentionWeights, V);

    return out;
  }

  /// Returns projection matrices [Wq], [Wk], and [Wv] as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['Wq'] = Wq.value;
    weightsMap['Wk'] = Wk.value;
    weightsMap['Wv'] = Wv.value;
    return weightsMap;
  }

  /// Sets projection matrices [Wq], [Wk], and [Wv] directly into their flat 1D data buffers from a map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      int idx = 0;
      for (int r = 0; r < newDataDynamic.length; r = r + 1) {
        List<dynamic> rowDynamic = newDataDynamic[r] as List<dynamic>;
        for (int c = 0; c < rowDynamic.length; c = c + 1) {
          tensor.data[idx] = rowDynamic[c] as double;
          idx = idx + 1;
        }
      }
    }

    copyMatrix(Wq, weightsMap['Wq'] as List<dynamic>);
    copyMatrix(Wk, weightsMap['Wk'] as List<dynamic>);
    copyMatrix(Wv, weightsMap['Wv'] as List<dynamic>);
  }
}
/*void main() {
  int dModel = 4;
  int seqLen = 2;

  Matrix inputData = [];
  for (int i = 0; i < seqLen; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < dModel; j = j + 1) {
      row.add((i + j + 1).toDouble());
    }
    inputData.add(row);
  }
  Tensor<Matrix> input = Tensor<Matrix>(inputData);

  SingleHeadAttention attention = SingleHeadAttention(dModel);
  attention.build(input);

  Matrix targetData = [];
  for (int i = 0; i < seqLen; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < dModel; j = j + 1) {
      row.add(0.0);
    }
    targetData.add(row);
  }
  Tensor<Matrix> target = Tensor<Matrix>(targetData);

  SGD optimizer = SGD(attention.parameters, learningRate: 0.01);

  for (int epoch = 0; epoch < 20; epoch = epoch + 1) {
    Tensor<Matrix> output = attention.forward(input);
    Tensor<Scalar> loss = mseMatrix(output, target);

    print(loss.value);

    loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
  }
}*/