import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class DualLSTMTL extends TapeLayer {
  @override
  String get name {
    return 'DualLSTMTapeLayer';
  }

  int hiddenSize;
  int lowerTierClockCycle;

  // Stored natively as [C, H] to avoid transposeGPU gradient loss
  late GPUTensor<Matrix> lW_f, lW_i, lW_c, lW_o;
  late GPUTensor<Matrix> lb_f, lb_i, lb_c, lb_o;

  late GPUTensor<Matrix> hW_f, hW_i, hW_c, hW_o;
  late GPUTensor<Matrix> hb_f, hb_i, hb_c, hb_o;

  DualLSTMTL(this.hiddenSize, {this.lowerTierClockCycle = 7});

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(<GPUTensor>[lW_f, lb_f, lW_i, lb_i, lW_c, lb_c, lW_o, lb_o]);
      params.addAll(<GPUTensor>[hW_f, hb_f, hW_i, hb_i, hW_c, hb_c, hW_o, hb_o]);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    int inputSize = typedInput.shape[1];

    int lowerCombinedSize = hiddenSize + hiddenSize + inputSize;
    int higherCombinedSize = hiddenSize + hiddenSize;

    Random random = Random();

    // Initialize as [fanIn, fanOut] so Row Matrix [1, fanIn] * [fanIn, fanOut] works natively
    List<List<double>> initWeightsTransposed(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      List<List<double>> values = <List<double>>[];
      for (int i = 0; i < fanIn; i = i + 1) { // Rows = fanIn
        List<double> row = <double>[];
        for (int j = 0; j < fanOut; j = j + 1) { // Cols = fanOut
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        values.add(row);
      }
      return values;
    }

    // Initialize as [1, size] Row Matrix
    List<List<double>> initBiasRow(int size) {
      List<double> values = <double>[];
      for (int i = 0; i < size; i = i + 1) {
        values.add(0.0);
      }
      return <List<double>>[values];
    }

    lW_f = GPUTensor<Matrix>(initWeightsTransposed(lowerCombinedSize, hiddenSize));
    lW_i = GPUTensor<Matrix>(initWeightsTransposed(lowerCombinedSize, hiddenSize));
    lW_c = GPUTensor<Matrix>(initWeightsTransposed(lowerCombinedSize, hiddenSize));
    lW_o = GPUTensor<Matrix>(initWeightsTransposed(lowerCombinedSize, hiddenSize));

    lb_f = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    lb_i = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    lb_c = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    lb_o = GPUTensor<Matrix>(initBiasRow(hiddenSize));

    hW_f = GPUTensor<Matrix>(initWeightsTransposed(higherCombinedSize, hiddenSize));
    hW_i = GPUTensor<Matrix>(initWeightsTransposed(higherCombinedSize, hiddenSize));
    hW_c = GPUTensor<Matrix>(initWeightsTransposed(higherCombinedSize, hiddenSize));
    hW_o = GPUTensor<Matrix>(initWeightsTransposed(higherCombinedSize, hiddenSize));

    hb_f = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    hb_i = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    hb_c = GPUTensor<Matrix>(initBiasRow(hiddenSize));
    hb_o = GPUTensor<Matrix>(initBiasRow(hiddenSize));

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    int totalSteps = typedInput.shape[0];
    int inputSize = typedInput.shape[1];

    List<double> zeros = <double>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      zeros.add(0.0);
    }
    List<List<double>> initialState = <List<double>>[zeros];

    GPUTensor<Matrix> lh = GPUTensor<Matrix>(initialState);
    GPUTensor<Matrix> lc = GPUTensor<Matrix>(initialState);
    GPUTensor<Matrix> hh = GPUTensor<Matrix>(initialState);
    GPUTensor<Matrix> hc = GPUTensor<Matrix>(initialState);
    intermediates.addAll(<GPUTensor>[lh, lc, hh, hc]);

    for (int i = 0; i < totalSteps; i = i + 1) {
      // 1. Safely slice the vector and reshape it into a [1, InputSize] Row Matrix
      GPUTensor<Vector> x_vec = selectRowGPU(typedInput, i, tape);
      GPUTensor<Matrix> x_t = reshapeVectorToMatrixGPU(x_vec, 1, inputSize, tape);
      intermediates.addAll(<GPUTensor>[x_vec, x_t]);

      // --- LOWER TIER ---
      // [1, H] + [1, H] + [1, I] = [1, C]
      GPUTensor<Matrix> comb_low = concatenateMatricesByColumnGPU(<GPUTensor<Matrix>>[lh, hc, x_t], tape);
      intermediates.add(comb_low);

      // [1, C] * [C, H] = [1, H] (Perfect Matrix Multiplication!)
      GPUTensor<Matrix> lf_linear = matMulGPU(comb_low, lW_f, tape);
      GPUTensor<Matrix> lf_biased = addMatrixGPU(lf_linear, lb_f, tape);
      GPUTensor<Matrix> lf_t = sigmoidMatrixGPU(lf_biased, tape);
      intermediates.addAll(<GPUTensor>[lf_linear, lf_biased, lf_t]);

      GPUTensor<Matrix> li_linear = matMulGPU(comb_low, lW_i, tape);
      GPUTensor<Matrix> li_biased = addMatrixGPU(li_linear, lb_i, tape);
      GPUTensor<Matrix> li_t = sigmoidMatrixGPU(li_biased, tape);
      intermediates.addAll(<GPUTensor>[li_linear, li_biased, li_t]);

      GPUTensor<Matrix> lc_tilde_linear = matMulGPU(comb_low, lW_c, tape);
      GPUTensor<Matrix> lc_tilde_biased = addMatrixGPU(lc_tilde_linear, lb_c, tape);
      GPUTensor<Matrix> lc_tilde = tanhMatrixGPU(lc_tilde_biased, tape);
      intermediates.addAll(<GPUTensor>[lc_tilde_linear, lc_tilde_biased, lc_tilde]);

      GPUTensor<Matrix> lc_retained = elementWiseMultiplyMatrixGPU(lf_t, lc, tape);
      GPUTensor<Matrix> lc_new_info = elementWiseMultiplyMatrixGPU(li_t, lc_tilde, tape);
      lc = addMatrixGPU(lc_retained, lc_new_info, tape);
      intermediates.addAll(<GPUTensor>[lc_retained, lc_new_info, lc]);

      GPUTensor<Matrix> lo_linear = matMulGPU(comb_low, lW_o, tape);
      GPUTensor<Matrix> lo_biased = addMatrixGPU(lo_linear, lb_o, tape);
      GPUTensor<Matrix> lo_t = sigmoidMatrixGPU(lo_biased, tape);
      GPUTensor<Matrix> lc_activated = tanhMatrixGPU(lc, tape);
      lh = elementWiseMultiplyMatrixGPU(lo_t, lc_activated, tape);
      intermediates.addAll(<GPUTensor>[lo_linear, lo_biased, lo_t, lc_activated, lh]);

      // --- HIGHER TIER ---
      if (i > 0 && (i + 1) % lowerTierClockCycle == 0) {
        GPUTensor<Matrix> comb_high = concatenateMatricesByColumnGPU(<GPUTensor<Matrix>>[hh, lh], tape);
        intermediates.add(comb_high);

        GPUTensor<Matrix> hf_linear = matMulGPU(comb_high, hW_f, tape);
        GPUTensor<Matrix> hf_biased = addMatrixGPU(hf_linear, hb_f, tape);
        GPUTensor<Matrix> hf_t = sigmoidMatrixGPU(hf_biased, tape);
        intermediates.addAll(<GPUTensor>[hf_linear, hf_biased, hf_t]);

        GPUTensor<Matrix> hi_linear = matMulGPU(comb_high, hW_i, tape);
        GPUTensor<Matrix> hi_biased = addMatrixGPU(hi_linear, hb_i, tape);
        GPUTensor<Matrix> hi_t = sigmoidMatrixGPU(hi_biased, tape);
        intermediates.addAll(<GPUTensor>[hi_linear, hi_biased, hi_t]);

        GPUTensor<Matrix> hc_tilde_linear = matMulGPU(comb_high, hW_c, tape);
        GPUTensor<Matrix> hc_tilde_biased = addMatrixGPU(hc_tilde_linear, hb_c, tape);
        GPUTensor<Matrix> hc_tilde = tanhMatrixGPU(hc_tilde_biased, tape);
        intermediates.addAll(<GPUTensor>[hc_tilde_linear, hc_tilde_biased, hc_tilde]);

        GPUTensor<Matrix> hc_retained = elementWiseMultiplyMatrixGPU(hf_t, hc, tape);
        GPUTensor<Matrix> hc_new_info = elementWiseMultiplyMatrixGPU(hi_t, hc_tilde, tape);
        hc = addMatrixGPU(hc_retained, hc_new_info, tape);
        intermediates.addAll(<GPUTensor>[hc_retained, hc_new_info, hc]);

        GPUTensor<Matrix> ho_linear = matMulGPU(comb_high, hW_o, tape);
        GPUTensor<Matrix> ho_biased = addMatrixGPU(ho_linear, hb_o, tape);
        GPUTensor<Matrix> ho_t = sigmoidMatrixGPU(ho_biased, tape);
        GPUTensor<Matrix> hc_activated = tanhMatrixGPU(hc, tape);
        hh = elementWiseMultiplyMatrixGPU(ho_t, hc_activated, tape);
        intermediates.addAll(<GPUTensor>[ho_linear, ho_biased, ho_t, hc_activated, hh]);
      }
    }

    // Unpack the final [1, H] Matrix back to a 1D Vector for standard mseGPU mapping!
    GPUTensor<Vector> finalVector = selectRowGPU(lh, 0, tape);
    intermediates.add(finalVector);

    return finalVector;
  }

  @override
  void free() {
    if (built) {
      lW_f.free(); lb_f.free(); lW_i.free(); lb_i.free();
      lW_c.free(); lb_c.free(); lW_o.free(); lb_o.free();
      hW_f.free(); hb_f.free(); hW_i.free(); hb_i.free();
      hW_c.free(); hb_c.free(); hW_o.free(); hb_o.free();
    }
  }

  // Flawlessly syncs with CPU structure by automatically transposing weights on load/save
  List<List<double>> _transposeForCpu(List<List<double>> m) {
    int rows = m.length;
    int cols = m[0].length;
    List<List<double>> res = <List<double>>[];
    for (int i = 0; i < cols; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < rows; j = j + 1) {
        row.add(m[j][i]);
      }
      res.add(row);
    }
    return res;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    lW_f.toCpu(); lb_f.toCpu(); lW_i.toCpu(); lb_i.toCpu();
    lW_c.toCpu(); lb_c.toCpu(); lW_o.toCpu(); lb_o.toCpu();
    hW_f.toCpu(); hb_f.toCpu(); hW_i.toCpu(); hb_i.toCpu();
    hW_c.toCpu(); hb_c.toCpu(); hW_o.toCpu(); hb_o.toCpu();

    wMap['lW_f'] = _transposeForCpu(lW_f.value); wMap['lW_i'] = _transposeForCpu(lW_i.value);
    wMap['lW_c'] = _transposeForCpu(lW_c.value); wMap['lW_o'] = _transposeForCpu(lW_o.value);
    wMap['hW_f'] = _transposeForCpu(hW_f.value); wMap['hW_i'] = _transposeForCpu(hW_i.value);
    wMap['hW_c'] = _transposeForCpu(hW_c.value); wMap['hW_o'] = _transposeForCpu(hW_o.value);

    wMap['lb_f'] = lb_f.value[0]; wMap['lb_i'] = lb_i.value[0];
    wMap['lb_c'] = lb_c.value[0]; wMap['lb_o'] = lb_o.value[0];
    wMap['hb_f'] = hb_f.value[0]; wMap['hb_i'] = hb_i.value[0];
    wMap['hb_c'] = hb_c.value[0]; wMap['hb_o'] = hb_o.value[0];

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) free();

    List<List<double>> extractAndTranspose(List<dynamic> raw) {
      List<List<double>> m = <List<double>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<double> row = <double>[];
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          row.add(rawRow[j] as double);
        }
        m.add(row);
      }
      return _transposeForCpu(m);
    }

    List<List<double>> wrapVectorToRow(List<dynamic> raw) {
      List<double> res = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        res.add(raw[i] as double);
      }
      return <List<double>>[res];
    }

    lW_f = GPUTensor<Matrix>(extractAndTranspose(newWeights['lW_f']!));
    lW_i = GPUTensor<Matrix>(extractAndTranspose(newWeights['lW_i']!));
    lW_c = GPUTensor<Matrix>(extractAndTranspose(newWeights['lW_c']!));
    lW_o = GPUTensor<Matrix>(extractAndTranspose(newWeights['lW_o']!));

    lb_f = GPUTensor<Matrix>(wrapVectorToRow(newWeights['lb_f']!));
    lb_i = GPUTensor<Matrix>(wrapVectorToRow(newWeights['lb_i']!));
    lb_c = GPUTensor<Matrix>(wrapVectorToRow(newWeights['lb_c']!));
    lb_o = GPUTensor<Matrix>(wrapVectorToRow(newWeights['lb_o']!));

    hW_f = GPUTensor<Matrix>(extractAndTranspose(newWeights['hW_f']!));
    hW_i = GPUTensor<Matrix>(extractAndTranspose(newWeights['hW_i']!));
    hW_c = GPUTensor<Matrix>(extractAndTranspose(newWeights['hW_c']!));
    hW_o = GPUTensor<Matrix>(extractAndTranspose(newWeights['hW_o']!));

    hb_f = GPUTensor<Matrix>(wrapVectorToRow(newWeights['hb_f']!));
    hb_i = GPUTensor<Matrix>(wrapVectorToRow(newWeights['hb_i']!));
    hb_c = GPUTensor<Matrix>(wrapVectorToRow(newWeights['hb_c']!));
    hb_o = GPUTensor<Matrix>(wrapVectorToRow(newWeights['hb_o']!));

    built = true;
  }
}