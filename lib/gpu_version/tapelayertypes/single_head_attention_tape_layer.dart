import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Applies Scaled Dot-Product Attention over a sequence.
/// Projects the input into Query, Key, and Value matrices, splits them into multiple heads,
/// calculates scaled dot-product attention, and projects the concatenated results back to the original dimension.
class SingleHeadAttentionTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'SingleHeadAttentionTapeLayer';

  int dModel;
  int dK;
  int dV;

  /// Learnable weight matrix to project the input into Queries.
  late GPUTensor<Matrix> wq;

  /// Learnable weight matrix to project the input into Keys.
  late GPUTensor<Matrix> wk;

  /// Learnable weight matrix to project the input into Values.
  late GPUTensor<Matrix> wv;

  /// A reference to the attention probabilities from the most recent forward pass.
  GPUTensor<Matrix>? lastAttentionWeights;

  /// Requires the model dimension [dModel]. Optionally accepts custom dimensions
  /// for the keys/queries [dK] and values [dV]. If omitted, they default to [dModel].
  SingleHeadAttentionTL(this.dModel, {int? dK, int? dV})
      : dK = dK ?? dModel,
        dV = dV ?? dModel;

  /// Returns the trainable projection matrices.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(wq);
      params.add(wk);
      params.add(wv);
    }
    return params;
  }

  /// Allocates VRAM for the Query, Key, and Value projection matrices.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
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

    wq = GPUTensor<Matrix>(initWeights(dModel, dK), allocTape,
        sourceOp: '$name:wq');
    wk = GPUTensor<Matrix>(initWeights(dModel, dK), allocTape,
        sourceOp: '$name:wk');
    wv = GPUTensor<Matrix>(initWeights(dModel, dV), allocTape,
        sourceOp: '$name:wv');

    built = true;
  }

  /// Appends the single-head attention operations to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    GPUTensor<Matrix> q = matMulGPU(input, wq, tape, allocTape);
    GPUTensor<Matrix> k = matMulGPU(input, wk, tape, allocTape);
    GPUTensor<Matrix> v = matMulGPU(input, wv, tape, allocTape);

    GPUTensor<Matrix> kT = transposeGPU(k, tape, allocTape);
    GPUTensor<Matrix> scores = matMulGPU(q, kT, tape, allocTape);

    double scaleFactor = 1.0 / sqrt(dK);
    GPUTensor<Matrix> scaledScores =
        scaleMatrixGPU(scores, scaleFactor, tape, allocTape);

    GPUTensor<Matrix> attentionWeights =
        softmaxMatrixGPU(scaledScores, tape, allocTape);
    lastAttentionWeights = attentionWeights;

    GPUTensor<Matrix> output = matMulGPU(attentionWeights, v, tape, allocTape);

    return output;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    wq.dataToCpu();
    wk.dataToCpu();
    wv.dataToCpu();

    wMap['Wq'] = wq.value;
    wMap['Wk'] = wk.value;
    wMap['Wv'] = wv.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> extractFlatMatrix(List<dynamic> raw) {
      List<double> flat = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          flat.add(rawRow[j] as double);
        }
      }
      return flat;
    }

    wq.pushData(extractFlatMatrix(newWeights['Wq']!));
    wk.pushData(extractFlatMatrix(newWeights['Wk']!));
    wv.pushData(extractFlatMatrix(newWeights['Wv']!));
  }
}
