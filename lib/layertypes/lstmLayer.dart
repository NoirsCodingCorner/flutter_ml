import 'dart:math';

import '../autogradEngine/tensor.dart';
import 'layer.dart';

class LSTMLayer extends Layer {
  @override
  String name = 'lstm';

  int hiddenSize;

  late Tensor<Matrix> W_f;
  late Tensor<Vector> b_f;

  late Tensor<Matrix> W_i;
  late Tensor<Vector> b_i;

  late Tensor<Matrix> W_c;
  late Tensor<Vector> b_c;

  late Tensor<Matrix> W_o;
  late Tensor<Vector> b_o;

  LSTMLayer(this.hiddenSize);

  @override
  List<Tensor> get parameters => [W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    int combinedSize = hiddenSize + inputSize;
    Random random = Random();

    Tensor<Matrix> initWeights(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      Matrix values = [];
      for (int i = 0; i < fanOut; i++) {
        Vector row = [];
        for (int j = 0; j < fanIn; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    W_f = initWeights(combinedSize, hiddenSize);
    W_i = initWeights(combinedSize, hiddenSize);
    W_c = initWeights(combinedSize, hiddenSize);
    W_o = initWeights(combinedSize, hiddenSize);

    b_f = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_i = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    b_o = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    Tensor<Vector> h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));
    Tensor<Vector> c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Vector timestep_x_list in sequence) {
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);
      Tensor<Vector> combined_input = concatenate(h, x_t);

      Tensor<Vector> f_t_linear = matVecMul(W_f, combined_input);
      Tensor<Vector> f_t_biased = addVector(f_t_linear, b_f);
      Tensor<Vector> f_t = sigmoid(f_t_biased);

      Tensor<Vector> i_t_linear = matVecMul(W_i, combined_input);
      Tensor<Vector> i_t_biased = addVector(i_t_linear, b_i);
      Tensor<Vector> i_t = sigmoid(i_t_biased);

      Tensor<Vector> c_tilde_t_linear = matVecMul(W_c, combined_input);
      Tensor<Vector> c_tilde_t_biased = addVector(c_tilde_t_linear, b_c);
      Tensor<Vector> c_tilde_t = vectorTanh(c_tilde_t_biased);

      Tensor<Vector> c_retained = elementWiseMultiply(f_t, c);
      Tensor<Vector> c_new_info = elementWiseMultiply(i_t, c_tilde_t);
      c = addVector(c_retained, c_new_info);

      Tensor<Vector> o_t_linear = matVecMul(W_o, combined_input);
      Tensor<Vector> o_t_biased = addVector(o_t_linear, b_o);
      Tensor<Vector> o_t = sigmoid(o_t_biased);

      Tensor<Vector> c_activated = vectorTanh(c);
      h = elementWiseMultiply(o_t, c_activated);
    }

    return h;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'W_f': W_f.value, 'b_f': b_f.value,
      'W_i': W_i.value, 'b_i': b_i.value,
      'W_c': W_c.value, 'b_c': b_c.value,
      'W_o': W_o.value, 'b_o': b_o.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void _copyMatrix(Tensor<Matrix> tensor, Matrix newData) {
      int height = tensor.value.length;
      int width = (height > 0) ? tensor.value[0].length : 0;
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          tensor.value[i][j] = newData[i][j];
        }
      }
    }

    void _copyVector(Tensor<Vector> tensor, List<dynamic> newData) {
      int length = tensor.value.length;
      for (int i = 0; i < length; i++) {
        tensor.value[i] = newData[i] as double;
      }
    }

    Matrix new_W_f = (weightsMap['W_f'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_f, new_W_f);
    _copyVector(b_f, weightsMap['b_f'] as List);

    Matrix new_W_i = (weightsMap['W_i'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_i, new_W_i);
    _copyVector(b_i, weightsMap['b_i'] as List);

    Matrix new_W_c = (weightsMap['W_c'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_c, new_W_c);
    _copyVector(b_c, weightsMap['b_c'] as List);

    Matrix new_W_o = (weightsMap['W_o'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_o, new_W_o);
    _copyVector(b_o, weightsMap['b_o'] as List);
  }
}