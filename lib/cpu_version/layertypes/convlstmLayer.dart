import 'dart:math';
import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

/// Applies a Convolutional Long Short-Term Memory (ConvLSTM) recurrent step over a sequence of 2D matrices formatted as a [Tensor3D].
/// Replaces standard matrix multiplications in an LSTM with 2D convolutions ([conv2d]) using 'same' padding to preserve spatial dimensions across recurrent state transitions.
/// Computes forget, input, cell, and output gates sequentially over the sequence length and returns the final hidden state [Matrix].
class ConvLSTMLayer extends Layer<Tensor3D, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'conv_lstm';

  /// Number of filter channels for the hidden state representation.
  int hiddenFilters;

  /// Spatial height and width of the 2D square convolution kernels.
  int kernelSize;

  /// Convolution kernel tensor for input-to-forget gate.
  late Tensor<Matrix> K_xf;

  /// Convolution kernel tensor for hidden-to-forget gate.
  late Tensor<Matrix> K_hf;

  /// Convolution kernel tensor for input-to-input gate.
  late Tensor<Matrix> K_xi;

  /// Convolution kernel tensor for hidden-to-input gate.
  late Tensor<Matrix> K_hi;

  /// Convolution kernel tensor for input-to-candidate cell state.
  late Tensor<Matrix> K_xc;

  /// Convolution kernel tensor for hidden-to-candidate cell state.
  late Tensor<Matrix> K_hc;

  /// Convolution kernel tensor for input-to-output gate.
  late Tensor<Matrix> K_xo;

  /// Convolution kernel tensor for hidden-to-output gate.
  late Tensor<Matrix> K_ho;

  /// Spatial bias matrix tensor for the forget gate.
  late Tensor<Matrix> b_f;

  /// Spatial bias matrix tensor for the input gate.
  late Tensor<Matrix> b_i;

  /// Spatial bias matrix tensor for the candidate cell state.
  late Tensor<Matrix> b_c;

  /// Spatial bias matrix tensor for the output gate.
  late Tensor<Matrix> b_o;

  /// Creates a [ConvLSTMLayer] with the specified [hiddenFilters] and [kernelSize].
  ConvLSTMLayer(this.hiddenFilters, this.kernelSize);

  /// Returns all trainable convolution kernels and bias matrices across all gates.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    params.add(K_xf);
    params.add(K_hf);
    params.add(b_f);
    params.add(K_xi);
    params.add(K_hi);
    params.add(b_i);
    params.add(K_xc);
    params.add(K_hc);
    params.add(b_c);
    params.add(K_xo);
    params.add(K_ho);
    params.add(b_o);
    return params;
  }

  /// Allocates and initializes convolution kernels with uniform scaling and biases to zero based on input spatial dimensions.
  @override
  void build(Tensor<Tensor3D> input) {
    Tensor3D inputSequence = input.value;
    int height = inputSequence[0].length;
    int width = inputSequence[0][0].length;
    Random random = Random();

    Tensor<Matrix> initKernel(int size) {
      double stddev = sqrt(1.0 / (size * size));
      Matrix values = [];
      for (int i = 0; i < size; i = i + 1) {
        Vector row = [];
        for (int j = 0; j < size; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    Tensor<Matrix> initBias() {
      Matrix biasValues = [];
      for (int i = 0; i < height; i = i + 1) {
        Vector row = [];
        for (int j = 0; j < width; j = j + 1) {
          row.add(0.0);
        }
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

  /// Executes the recurrent ConvLSTM forward pass on the CPU across all sequence frames and returns the final hidden state [Matrix].
  @override
  Tensor<Matrix> forward(Tensor<Tensor3D> input) {
    Tensor3D sequence = input.value;
    int seqLength = sequence.length;
    int height = sequence[0].length;
    int width = sequence[0][0].length;

    Matrix zeroMatrixH = [];
    Matrix zeroMatrixC = [];
    for (int i = 0; i < height; i = i + 1) {
      Vector rowH = [];
      Vector rowC = [];
      for (int j = 0; j < width; j = j + 1) {
        rowH.add(0.0);
        rowC.add(0.0);
      }
      zeroMatrixH.add(rowH);
      zeroMatrixC.add(rowC);
    }

    Tensor<Matrix> h = Tensor<Matrix>(zeroMatrixH);
    Tensor<Matrix> c = Tensor<Matrix>(zeroMatrixC);

    for (int t = 0; t < seqLength; t = t + 1) {
      Tensor<Matrix> xT = Tensor<Matrix>(sequence[t]);

      // Note: Swapped the order of arguments for conv2d here to match
      // the signature from tensor_math_cpu.dart: conv2d(input, kernel)
      Tensor<Matrix> fTInputconv = conv2d(xT, K_xf, padding: 'same');
      Tensor<Matrix> fTHiddenconv = conv2d(h, K_hf, padding: 'same');
      Tensor<Matrix> fTSum = addMatrix(fTInputconv, fTHiddenconv);
      Tensor<Matrix> fTBiased = addMatrix(fTSum, b_f);
      Tensor<Matrix> fT = sigmoidMatrix(fTBiased);

      Tensor<Matrix> iTInputconv = conv2d(xT, K_xi, padding: 'same');
      Tensor<Matrix> iTHiddenconv = conv2d(h, K_hi, padding: 'same');
      Tensor<Matrix> iTSum = addMatrix(iTInputconv, iTHiddenconv);
      Tensor<Matrix> iTBiased = addMatrix(iTSum, b_i);
      Tensor<Matrix> iT = sigmoidMatrix(iTBiased);

      Tensor<Matrix> cTildeTInputconv = conv2d(xT, K_xc, padding: 'same');
      Tensor<Matrix> cTildeTHiddenconv = conv2d(h, K_hc, padding: 'same');
      Tensor<Matrix> cTildeTSum = addMatrix(cTildeTInputconv, cTildeTHiddenconv);
      Tensor<Matrix> cTildeTBiased = addMatrix(cTildeTSum, b_c);
      Tensor<Matrix> cTildeT = tanhMatrix(cTildeTBiased);

      Tensor<Matrix> cRetained = elementWiseMultiplyMatrix(fT, c);
      Tensor<Matrix> cNewInfo = elementWiseMultiplyMatrix(iT, cTildeT);
      c = addMatrix(cRetained, cNewInfo);

      Tensor<Matrix> oTInputconv = conv2d(xT, K_xo, padding: 'same');
      Tensor<Matrix> oTHiddenconv = conv2d(h, K_ho, padding: 'same');
      Tensor<Matrix> oTSum = addMatrix(oTInputconv, oTHiddenconv);
      Tensor<Matrix> oTBiased = addMatrix(oTSum, b_o);
      Tensor<Matrix> oT = sigmoidMatrix(oTBiased);

      Tensor<Matrix> cActivated = tanhMatrix(c);
      h = elementWiseMultiplyMatrix(oT, cActivated);
    }

    return h;
  }

  /// Returns all convolution kernels and bias matrices as a map.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weightsMap = {};
    weightsMap['K_xf'] = K_xf.value;
    weightsMap['K_hf'] = K_hf.value;
    weightsMap['b_f']  = b_f.value;
    weightsMap['K_xi'] = K_xi.value;
    weightsMap['K_hi'] = K_hi.value;
    weightsMap['b_i']  = b_i.value;
    weightsMap['K_xc'] = K_xc.value;
    weightsMap['K_hc'] = K_hc.value;
    weightsMap['b_c']  = b_c.value;
    weightsMap['K_xo'] = K_xo.value;
    weightsMap['K_ho'] = K_ho.value;
    weightsMap['b_o']  = b_o.value;
    return weightsMap;
  }

  /// Sets all convolution kernels and bias matrices from a provided weights map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    void copyMatrix(Tensor<Matrix> tensor, List<dynamic> newDataDynamic) {
      int height = newDataDynamic.length;
      int width = 0;
      if (height > 0) {
        List<dynamic> firstRow = newDataDynamic[0] as List<dynamic>;
        width = firstRow.length;
      }

      for (int i = 0; i < height; i = i + 1) {
        List<dynamic> rowDynamic = newDataDynamic[i] as List<dynamic>;
        for (int j = 0; j < width; j = j + 1) {
          tensor.value[i][j] = rowDynamic[j] as double;
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