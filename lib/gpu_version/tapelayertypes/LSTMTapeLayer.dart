import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class LSTMTL extends TapeLayer {
  int hiddenSize;
  double gradClipValue;

  late GPUTensor<Matrix> W_xf; late GPUTensor<Matrix> W_hf; late GPUTensor<Matrix> b_f;
  late GPUTensor<Matrix> W_xi; late GPUTensor<Matrix> W_hi; late GPUTensor<Matrix> b_i;
  late GPUTensor<Matrix> W_xc; late GPUTensor<Matrix> W_hc; late GPUTensor<Matrix> b_c;
  late GPUTensor<Matrix> W_xo; late GPUTensor<Matrix> W_ho; late GPUTensor<Matrix> b_o;

  LSTMTL(this.hiddenSize, {this.gradClipValue = 1.0});

  @override
  String get name => 'LSTMTapeLayer';

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll([W_xf, W_hf, b_f, W_xi, W_hi, b_i, W_xc, W_hc, b_c, W_xo, W_ho, b_o]);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    GPUTensor<Matrix> sequence = input as GPUTensor<Matrix>;
    int inputSize = sequence.shape[1];
    Random random = Random();

    List<List<double>> initWeights(int fanIn, int fanOut) {
      double scale = sqrt(2.0 / (fanIn + fanOut));
      List<List<double>> values = <List<double>>[];
      for (int i = 0; i < fanIn; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < fanOut; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * scale);
        }
        values.add(row);
      }
      return values;
    }

    List<List<double>> initBias() {
      List<List<double>> values = <List<double>>[];
      List<double> row = <double>[];
      for (int i = 0; i < hiddenSize; i = i + 1) row.add(0.0);
      values.add(row);
      return values;
    }

    W_xf = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize));
    W_hf = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize));
    W_xi = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize));
    W_hi = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize));
    W_xc = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize));
    W_hc = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize));
    W_xo = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize));
    W_ho = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize));

    b_f = GPUTensor<Matrix>(initBias());
    b_i = GPUTensor<Matrix>(initBias());
    b_c = GPUTensor<Matrix>(initBias());
    b_o = GPUTensor<Matrix>(initBias());

    built = true;
  }

  // Helper to strictly slice a sequence row into a 1xN Matrix
  GPUTensor<Matrix> _sliceRowToMatrix(GPUTensor<Matrix> sequence, int rowIdx, CommandBuffer tape) {
    GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[1, sequence.shape[1]]);
    tape.putInt(OP_SLICE_ROW);
    tape.putString(sequence.id);
    tape.putString(out.id);
    tape.putInt(rowIdx);

    out.creator = GPUNode(
      <GPUTensor>[sequence],
          (CommandBuffer bTape) {
        bTape.putInt(OP_SLICE_ROW_BACKWARD);
        bTape.putString(out.id + '_grad');
        bTape.putString(sequence.id + '_grad');
        bTape.putInt(rowIdx);
      },
      opName: 'slice_row_matrix_gpu',
    );
    return out;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> sequence = input as GPUTensor<Matrix>;
    int seqLength = sequence.shape[0];

    List<List<double>> zeroMatrix = <List<double>>[];
    List<double> zeroRow = <double>[];
    for (int i = 0; i < hiddenSize; i = i + 1) zeroRow.add(0.0);
    zeroMatrix.add(zeroRow);

    GPUTensor<Matrix> h = GPUTensor<Matrix>(zeroMatrix);
    GPUTensor<Matrix> c = GPUTensor<Matrix>(zeroMatrix);

    intermediates.add(h);
    intermediates.add(c);

    for (int t = 0; t < seqLength; t = t + 1) {
      GPUTensor<Matrix> x_t = _sliceRowToMatrix(sequence, t, tape);
      intermediates.add(x_t);

      // Forget Gate
      GPUTensor<Matrix> f_t_x = matMulGPU(x_t, W_xf, tape);
      GPUTensor<Matrix> f_t_h = matMulGPU(h, W_hf, tape);
      GPUTensor<Matrix> f_t_sum = addMatrixGPU(f_t_x, f_t_h, tape);
      GPUTensor<Matrix> f_t_biased = addMatrixGPU(f_t_sum, b_f, tape);
      GPUTensor<Matrix> f_t = sigmoidMatrixGPU(f_t_biased, tape);

      intermediates.addAll([f_t_x, f_t_h, f_t_sum, f_t_biased, f_t]);

      // Input Gate
      GPUTensor<Matrix> i_t_x = matMulGPU(x_t, W_xi, tape);
      GPUTensor<Matrix> i_t_h = matMulGPU(h, W_hi, tape);
      GPUTensor<Matrix> i_t_sum = addMatrixGPU(i_t_x, i_t_h, tape);
      GPUTensor<Matrix> i_t_biased = addMatrixGPU(i_t_sum, b_i, tape);
      GPUTensor<Matrix> i_t = sigmoidMatrixGPU(i_t_biased, tape);

      intermediates.addAll([i_t_x, i_t_h, i_t_sum, i_t_biased, i_t]);

      // Cell Candidate
      GPUTensor<Matrix> c_tilde_t_x = matMulGPU(x_t, W_xc, tape);
      GPUTensor<Matrix> c_tilde_t_h = matMulGPU(h, W_hc, tape);
      GPUTensor<Matrix> c_tilde_t_sum = addMatrixGPU(c_tilde_t_x, c_tilde_t_h, tape);
      GPUTensor<Matrix> c_tilde_t_biased = addMatrixGPU(c_tilde_t_sum, b_c, tape);
      GPUTensor<Matrix> c_tilde_t = tanhMatrixGPU(c_tilde_t_biased, tape);

      intermediates.addAll([c_tilde_t_x, c_tilde_t_h, c_tilde_t_sum, c_tilde_t_biased, c_tilde_t]);

      // Cell State Update
      GPUTensor<Matrix> c_retained = elementWiseMultiplyMatrixGPU(f_t, c, tape);
      GPUTensor<Matrix> c_new_info = elementWiseMultiplyMatrixGPU(i_t, c_tilde_t, tape);
      c = addMatrixGPU(c_retained, c_new_info, tape);

      intermediates.addAll([c_retained, c_new_info]);
      if (t < seqLength - 1) intermediates.add(c);

      // Output Gate
      GPUTensor<Matrix> o_t_x = matMulGPU(x_t, W_xo, tape);
      GPUTensor<Matrix> o_t_h = matMulGPU(h, W_ho, tape);
      GPUTensor<Matrix> o_t_sum = addMatrixGPU(o_t_x, o_t_h, tape);
      GPUTensor<Matrix> o_t_biased = addMatrixGPU(o_t_sum, b_o, tape);
      GPUTensor<Matrix> o_t = sigmoidMatrixGPU(o_t_biased, tape);

      intermediates.addAll([o_t_x, o_t_h, o_t_sum, o_t_biased, o_t]);

      // Hidden State Update
      GPUTensor<Matrix> c_activated = tanhMatrixGPU(c, tape);
      h = elementWiseMultiplyMatrixGPU(o_t, c_activated, tape);

      intermediates.add(c_activated);
      if (t < seqLength - 1) intermediates.add(h);
    }

    // Inject Gradient Clipping
    if (h.creator != null) {
      void Function(CommandBuffer) originalBackward = h.creator!.backwardFn;
      h.creator!.backwardFn = (CommandBuffer bTape) {
        originalBackward(bTape);
        for (int p = 0; p < parameters.length; p = p + 1) {
          bTape.putInt(OP_CLIP_GRAD_VALUE);
          bTape.putString(parameters[p].id + '_grad');
          bTape.putFloat(gradClipValue);
        }
      };
    }

    return h;
  }

  @override
  void free() {
    if (built) {
      for (int p = 0; p < parameters.length; p = p + 1) parameters[p].free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() => {};
  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}