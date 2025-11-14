import 'dart:math';

import '../activationFunctions/softmax.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

class SingleHeadAttention extends Layer {
  @override
  String name = 'single_head_attention';
  int dModel;
  int dK;
  int dV;

  late Tensor<Matrix> Wq;
  late Tensor<Matrix> Wk;
  late Tensor<Matrix> Wv;

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

    lastAttentionWeights = attentionWeights;

    Tensor<Matrix> out = matMul(attentionWeights, V);

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'Wq': Wq.value,
      'Wk': Wk.value,
      'Wv': Wv.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void _copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      Matrix newData = newDataDynamic.map((dynamic row) {
        return (row as List<dynamic>).map((dynamic val) => val as double).toList();
      }).toList();

      int height = tensor.value.length;
      int width = (height > 0) ? tensor.value[0].length : 0;
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          tensor.value[i][j] = newData[i][j];
        }
      }
    }

    _copyMatrix(Wq, weightsMap['Wq'] as List<dynamic>);
    _copyMatrix(Wk, weightsMap['Wk'] as List<dynamic>);
    _copyMatrix(Wv, weightsMap['Wv'] as List<dynamic>);
  }
}