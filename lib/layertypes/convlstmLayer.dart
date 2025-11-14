import 'dart:math';

import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import '../optimizers/sgd.dart';
import 'denseLayer.dart';
import 'flattenLayer.dart';
import 'layer.dart';

class ConvLSTMLayer extends Layer {
  @override
  String name = 'conv_lstm';
  int hiddenFilters;
  int kernelSize;

  late Tensor<Matrix> K_xf, K_hf;
  late Tensor<Matrix> K_xi, K_hi;
  late Tensor<Matrix> K_xc, K_hc;
  late Tensor<Matrix> K_xo, K_ho;
  late Tensor<Matrix> b_f, b_i, b_c, b_o;

  ConvLSTMLayer(this.hiddenFilters, this.kernelSize);

  @override
  List<Tensor> get parameters => [
    K_xf, K_hf, b_f,
    K_xi, K_hi, b_i,
    K_xc, K_hc, b_c,
    K_xo, K_ho, b_o,
  ];

  @override
  void build(Tensor<dynamic> input) {
    Tensor3D inputSequence = input.value as Tensor3D;
    int height = inputSequence[0].length;
    int width = inputSequence[0][0].length;
    Random random = Random();

    Tensor<Matrix> initKernel(int size) {
      double stddev = sqrt(1.0 / (size * size));
      Matrix values = [];
      for (int i = 0; i < size; i++) {
        Vector row = [];
        for (int j = 0; j < size; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    Tensor<Matrix> initBias() {
      Matrix biasValues = [];
      for (int i = 0; i < height; i++) {
        Vector row = List<double>.filled(width, 0.0);
        biasValues.add(row);
      }
      return Tensor<Matrix>(biasValues);
    }

    K_xf = initKernel(kernelSize);
    K_hf = initKernel(kernelSize);
    K_xi = initKernel(kernelSize);
    K_hi = initKernel(kernelSize);
    K_xc = initKernel(kernelSize);
    K_hc = initKernel(kernelSize);
    K_xo = initKernel(kernelSize);
    K_ho = initKernel(kernelSize);

    b_f = initBias();
    b_i = initBias();
    b_c = initBias();
    b_o = initBias();

    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor3D sequence = (input as Tensor<Tensor3D>).value;
    int height = sequence[0].length;
    int width = sequence[0][0].length;

    Matrix zeroMatrix = [];
    for (int i = 0; i < height; i++) {
      zeroMatrix.add(List<double>.filled(width, 0.0));
    }
    Tensor<Matrix> h = Tensor<Matrix>(zeroMatrix);
    Tensor<Matrix> c = Tensor<Matrix>(zeroMatrix);

    for (Matrix timestep_x_matrix in sequence) {
      Tensor<Matrix> x_t = Tensor<Matrix>(timestep_x_matrix);

      Tensor<Matrix> f_t_inputConv = conv2d(K_xf, x_t, padding: 'same');
      Tensor<Matrix> f_t_hiddenConv = conv2d(K_hf, h, padding: 'same');
      Tensor<Matrix> f_t_sum = addMatrix(f_t_inputConv, f_t_hiddenConv);
      Tensor<Matrix> f_t_biased = addMatrix(f_t_sum, b_f);
      Tensor<Matrix> f_t = sigmoidMatrix(f_t_biased);

      Tensor<Matrix> i_t_inputConv = conv2d(K_xi, x_t, padding: 'same');
      Tensor<Matrix> i_t_hiddenConv = conv2d(K_hi, h, padding: 'same');
      Tensor<Matrix> i_t_sum = addMatrix(i_t_inputConv, i_t_hiddenConv);
      Tensor<Matrix> i_t_biased = addMatrix(i_t_sum, b_i);
      Tensor<Matrix> i_t = sigmoidMatrix(i_t_biased);

      Tensor<Matrix> c_tilde_t_inputConv = conv2d(K_xc, x_t, padding: 'same');
      Tensor<Matrix> c_tilde_t_hiddenConv = conv2d(K_hc, h, padding: 'same');
      Tensor<Matrix> c_tilde_t_sum = addMatrix(c_tilde_t_inputConv, c_tilde_t_hiddenConv);
      Tensor<Matrix> c_tilde_t_biased = addMatrix(c_tilde_t_sum, b_c);
      Tensor<Matrix> c_tilde_t = tanhMatrix(c_tilde_t_biased);

      Tensor<Matrix> c_retained = elementWiseMultiplyMatrix(f_t, c);
      Tensor<Matrix> c_new_info = elementWiseMultiplyMatrix(i_t, c_tilde_t);
      c = addMatrix(c_retained, c_new_info);

      Tensor<Matrix> o_t_inputConv = conv2d(K_xo, x_t, padding: 'same');
      Tensor<Matrix> o_t_hiddenConv = conv2d(K_ho, h, padding: 'same');
      Tensor<Matrix> o_t_sum = addMatrix(o_t_inputConv, o_t_hiddenConv);
      Tensor<Matrix> o_t_biased = addMatrix(o_t_sum, b_o);
      Tensor<Matrix> o_t = sigmoidMatrix(o_t_biased);

      Tensor<Matrix> c_activated = tanhMatrix(c);
      h = elementWiseMultiplyMatrix(o_t, c_activated);
    }

    return h;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {
      'K_xf': K_xf.value,
      'K_hf': K_hf.value,
      'b_f':  b_f.value,
      'K_xi': K_xi.value,
      'K_hi': K_hi.value,
      'b_i':  b_i.value,
      'K_xc': K_xc.value,
      'K_hc': K_hc.value,
      'b_c':  b_c.value,
      'K_xo': K_xo.value,
      'K_ho': K_ho.value,
      'b_o':  b_o.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
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

    copyMatrix(K_xf, weightsMap['K_xf'] as List<dynamic>);
    copyMatrix(K_hf, weightsMap['K_hf'] as List<dynamic>);
    copyMatrix(b_f,  weightsMap['b_f']  as List<dynamic>);

    copyMatrix(K_xi, weightsMap['K_xi'] as List<dynamic>);
    copyMatrix(K_hi, weightsMap['K_hi'] as List<dynamic>);
    copyMatrix(b_i,  weightsMap['b_i']  as List<dynamic>);

    copyMatrix(K_xc, weightsMap['K_xc'] as List<dynamic>);
    copyMatrix(K_hc, weightsMap['K_hc'] as List<dynamic>);
    copyMatrix(b_c,  weightsMap['b_c']  as List<dynamic>);

    copyMatrix(K_xo, weightsMap['K_xo'] as List<dynamic>);
    copyMatrix(K_ho, weightsMap['K_ho'] as List<dynamic>);
    copyMatrix(b_o,  weightsMap['b_o']  as List<dynamic>);
  }
}