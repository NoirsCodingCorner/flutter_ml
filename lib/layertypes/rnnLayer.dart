import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import 'layer.dart';

class RNN extends Layer {
  @override
  String name = 'rnn';

  int hiddenSize;
  ActivationFunction activation;

  late Tensor<Matrix> W_xh;
  late Tensor<Matrix> W_hh;
  late Tensor<Vector> b_h;

  RNN(this.hiddenSize, {required this.activation});

  @override
  List<Tensor> get parameters => [W_xh, W_hh, b_h];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    double xavierStdDev(int fanIn, int fanOut) => sqrt(2.0 / (fanIn + fanOut));

    double inputToHiddenStdDev = xavierStdDev(inputSize, hiddenSize);
    Matrix wXhValues = [];
    for (int i = 0; i < hiddenSize; i++) {
      Vector row = [];
      for (int j = 0; j < inputSize; j++) {
        row.add((random.nextDouble() * 2 - 1) * inputToHiddenStdDev);
      }
      wXhValues.add(row);
    }

    double hiddenToHiddenStdDev = xavierStdDev(hiddenSize, hiddenSize);
    Matrix wHhValues = [];
    for (int i = 0; i < hiddenSize; i++) {
      Vector row = [];
      for (int j = 0; j < hiddenSize; j++) {
        row.add((random.nextDouble() * 2 - 1) * hiddenToHiddenStdDev);
      }
      wHhValues.add(row);
    }

    W_xh = Tensor<Matrix>(wXhValues);
    W_hh = Tensor<Matrix>(wHhValues);
    b_h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix sequence = (input as Tensor<Matrix>).value;
    Tensor<Vector> h = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Vector timestep_x_list in sequence) {
      Tensor<Vector> x_t = Tensor<Vector>(timestep_x_list);

      Tensor<Vector> inputPart = matVecMul(W_xh, x_t);
      Tensor<Vector> hiddenPart = matVecMul(W_hh, h);
      Tensor<Vector> combined = addVector(addVector(inputPart, hiddenPart), b_h);

      h = activation.call(combined) as Tensor<Vector>;
    }

    return h;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'W_xh': W_xh.value,
      'W_hh': W_hh.value,
      'b_h': b_h.value,
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

    Matrix new_W_xh = (weightsMap['W_xh'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_xh, new_W_xh);

    Matrix new_W_hh = (weightsMap['W_hh'] as List<dynamic>).map((dynamic row) {
      return (row as List<dynamic>).map((dynamic val) => val as double).toList();
    }).toList();
    _copyMatrix(W_hh, new_W_hh);

    _copyVector(b_h, weightsMap['b_h'] as List);
  }
}