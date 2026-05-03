import 'dart:math';

import '../optimizers/sgd.dart';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class SingleHeadAttention extends Layer<Matrix, Matrix> {
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
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(Wq);
    params.add(Wk);
    params.add(Wv);
    return params;
  }

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

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['Wq'] = Wq.value;
    weightsMap['Wk'] = Wk.value;
    weightsMap['Wv'] = Wv.value;
    return weightsMap;
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void _copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      int idx = 0;
      for (int r = 0; r < newDataDynamic.length; r = r + 1) {
        List<dynamic> rowDynamic = newDataDynamic[r] as List<dynamic>;
        for (int c = 0; c < rowDynamic.length; c = c + 1) {
          tensor.data[idx] = rowDynamic[c] as double;
          idx = idx + 1;
        }
      }
    }

    _copyMatrix(Wq, weightsMap['Wq'] as List<dynamic>);
    _copyMatrix(Wk, weightsMap['Wk'] as List<dynamic>);
    _copyMatrix(Wv, weightsMap['Wv'] as List<dynamic>);
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