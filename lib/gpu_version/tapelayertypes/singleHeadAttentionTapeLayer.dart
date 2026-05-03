import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class SingleHeadAttentionTL extends TapeLayer {
  @override
  String get name {
    return 'SingleHeadAttentionTapeLayer';
  }

  int dModel;
  int dK;
  int dV;

  late GPUTensor<Matrix> Wq;
  late GPUTensor<Matrix> Wk;
  late GPUTensor<Matrix> Wv;

  GPUTensor<Matrix>? lastAttentionWeights;

  SingleHeadAttentionTL(this.dModel, {int? dK, int? dV})
      : dK = dK ?? dModel,
        dV = dV ?? dModel;

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(Wq);
      params.add(Wk);
      params.add(Wv);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    Random random = Random();

    List<List<double>> initWeights(int rows, int cols) {
      double stddev = sqrt(1.0 / rows);
      List<List<double>> values = <List<double>>[];
      for (int i = 0; i < rows; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < cols; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        values.add(row);
      }
      return values;
    }

    Wq = GPUTensor<Matrix>(initWeights(dModel, dK));
    Wk = GPUTensor<Matrix>(initWeights(dModel, dK));
    Wv = GPUTensor<Matrix>(initWeights(dModel, dV));

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> x = input as GPUTensor<Matrix>;

    GPUTensor<Matrix> q = matMulGPU(x, Wq, tape);
    GPUTensor<Matrix> k = matMulGPU(x, Wk, tape);
    GPUTensor<Matrix> v = matMulGPU(x, Wv, tape);

    intermediates.add(q);
    intermediates.add(k);
    intermediates.add(v);

    GPUTensor<Matrix> k_t = transposeGPU(k, tape);
    GPUTensor<Matrix> scores = matMulGPU(q, k_t, tape);

    intermediates.add(k_t);
    intermediates.add(scores);

    double scaleFactor = 1.0 / sqrt(dK);
    GPUTensor<Matrix> scaledScores = scaleMatrixGPU(scores, scaleFactor, tape);
    intermediates.add(scaledScores);

    GPUTensor<Matrix> attentionWeights = softmaxMatrixGPU(scaledScores, tape);
    lastAttentionWeights = attentionWeights;
    intermediates.add(attentionWeights);

    GPUTensor<Matrix> output = matMulGPU(attentionWeights, v, tape);

    // ⚡ FIXED: Ensure the final output is registered for zeroing
    intermediates.add(output);

    return output;
  }

  @override
  void free() {
    if (built) {
      Wq.free();
      Wk.free();
      Wv.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    Wq.toCpu();
    Wk.toCpu();
    Wv.toCpu();

    wMap['Wq'] = Wq.value;
    wMap['Wk'] = Wk.value;
    wMap['Wv'] = Wv.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      free();
    }

    List<List<double>> extractMatrix(List<dynamic> raw) {
      List<List<double>> result = <List<double>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<double> row = <double>[];
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          row.add(rawRow[j] as double);
        }
        result.add(row);
      }
      return result;
    }

    Wq = GPUTensor<Matrix>(extractMatrix(newWeights['Wq']!));
    Wk = GPUTensor<Matrix>(extractMatrix(newWeights['Wk']!));
    Wv = GPUTensor<Matrix>(extractMatrix(newWeights['Wv']!));

    built = true;
  }
}