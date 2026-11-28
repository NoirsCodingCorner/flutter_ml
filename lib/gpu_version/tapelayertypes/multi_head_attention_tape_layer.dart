import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Applies Multi-Head Attention to a sequence.
/// Projects the input into Query, Key, and Value matrices, splits them into multiple heads,
/// calculates scaled dot-product attention, and projects the concatenated results back to the original dimension.
class MultiHeadAttentionTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'MultiHeadAttentionTapeLayer';

  int dModel;
  int numHeads;
  late int dHead;

  // The 8 Monolithic Parameters for transformer-style attention
  late GPUTensor<Matrix> wq;
  late GPUTensor<Vector> bq;
  late GPUTensor<Matrix> wk;
  late GPUTensor<Vector> bk;
  late GPUTensor<Matrix> wv;
  late GPUTensor<Vector> bv;
  late GPUTensor<Matrix> wo;
  late GPUTensor<Vector> bo;

  /// Optional mask added to the attention scores before the softmax operation.
  GPUTensor<Vector>? attentionMask;

  /// Requires the model dimension [dModel] and the number of attention heads [numHeads].
  /// [dModel] must be cleanly divisible by [numHeads].
  MultiHeadAttentionTL(this.dModel, this.numHeads) {
    dHead = dModel ~/ numHeads;
  }

  /// Returns the 8 trainable weight and bias matrices.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(<GPUTensor>[wq, bq, wk, bk, wv, bv, wo, bo]);
    }
    return params;
  }

  /// Allocates VRAM for the Query, Key, Value, and Output projections.
  /// Uses a scaled random uniform distribution for initial weights and zeros for biases.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    double scale = sqrt(2.0 / (dModel + dModel));

    wq = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale, allocTape,
        sourceOp: '$name:wq');
    bq = GPUTensor<Vector>.empty(<int>[dModel], allocTape,
        sourceOp: '$name:bq');

    wk = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale, allocTape,
        sourceOp: '$name:wk');
    bk = GPUTensor<Vector>.empty(<int>[dModel], allocTape,
        sourceOp: '$name:bk');

    wv = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale, allocTape,
        sourceOp: '$name:wv');
    bv = GPUTensor<Vector>.empty(<int>[dModel], allocTape,
        sourceOp: '$name:bv');

    wo = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale, allocTape,
        sourceOp: '$name:wo');
    bo = GPUTensor<Vector>.empty(<int>[dModel], allocTape,
        sourceOp: '$name:bo');

    built = true;
  }

  /// Used specifically for SafeTensors compatibility to map weights to standard transformer nomenclature.
  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      map['$prefix.self.query.weight'] = wq;
      map['$prefix.self.query.bias'] = bq;

      map['$prefix.self.key.weight'] = wk;
      map['$prefix.self.key.bias'] = bk;

      map['$prefix.self.value.weight'] = wv;
      map['$prefix.self.value.bias'] = bv;

      map['$prefix.output.dense.weight'] = wo;
      map['$prefix.output.dense.bias'] = bo;
    }
    return map;
  }

  /// Appends the multi-head attention operations to the [tape].
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    bool hasMask = (attentionMask != null);

    // 1. Compute Full Q, K, V
    GPUTensor<Matrix> qMatMul = matMulGPU(input, wq, tape, allocTape);
    GPUTensor<Matrix> Q = addBiasToMatMulOutGPU(qMatMul, bq, tape, allocTape);

    GPUTensor<Matrix> kMatMul = matMulGPU(input, wk, tape, allocTape);
    GPUTensor<Matrix> K = addBiasToMatMulOutGPU(kMatMul, bk, tape, allocTape);

    GPUTensor<Matrix> vMatMul = matMulGPU(input, wv, tape, allocTape);
    GPUTensor<Matrix> V = addBiasToMatMulOutGPU(vMatMul, bv, tape, allocTape);

    List<GPUTensor<Matrix>> headOutputs = <GPUTensor<Matrix>>[];
    double scaleFactor = 1.0 / sqrt(dHead);

    // 2. Slice and process each head independently
    for (int i = 0; i < numHeads; i = i + 1) {
      int startCol = i * dHead;
      int endCol = startCol + dHead;

      GPUTensor<Matrix> qHead =
      sliceColumnGPU(Q, startCol, endCol, tape, allocTape);
      GPUTensor<Matrix> kHead =
      sliceColumnGPU(K, startCol, endCol, tape, allocTape);
      GPUTensor<Matrix> vHead =
      sliceColumnGPU(V, startCol, endCol, tape, allocTape);

      // Attention Equation: Softmax(Q * K^T / sqrt(dHead)) * V
      GPUTensor<Matrix> kHeadT = transposeGPU(kHead, tape, allocTape);
      GPUTensor<Matrix> scores = matMulGPU(qHead, kHeadT, tape, allocTape);
      GPUTensor<Matrix> scaledScores =
      scaleMatrixGPU(scores, scaleFactor, tape, allocTape);

      GPUTensor<Matrix> maskedScores;
      if (hasMask) {
        maskedScores = broadcastAddVectorToMatrixGPU(
            scaledScores, attentionMask!, tape, allocTape);
      } else {
        maskedScores = scaledScores;
      }

      GPUTensor<Matrix> probs =
      softmaxMatrixGPU(maskedScores, tape, allocTape);
      GPUTensor<Matrix> headOut = matMulGPU(probs, vHead, tape, allocTape);

      headOutputs.add(headOut);
    }

    // 3. Concatenate all heads horizontally
    GPUTensor<Matrix> concatOut =
    concatenateMatricesByColumnGPU(headOutputs, tape, allocTape);

    // 4. Final Output Projection
    GPUTensor<Matrix> oMatMul = matMulGPU(concatOut, wo, tape, allocTape);
    GPUTensor<Matrix> finalOut =
    addBiasToMatMulOutGPU(oMatMul, bo, tape, allocTape);

    return finalOut;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    wq.dataToCpu();
    bq.dataToCpu();
    wk.dataToCpu();
    bk.dataToCpu();
    wv.dataToCpu();
    bv.dataToCpu();
    wo.dataToCpu();
    bo.dataToCpu();

    wMap['Wq'] = wq.value;
    wMap['bq'] = bq.value;
    wMap['Wk'] = wk.value;
    wMap['bk'] = bk.value;
    wMap['Wv'] = wv.value;
    wMap['bv'] = bv.value;
    wMap['Wo'] = wo.value;
    wMap['bo'] = bo.value;

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

    List<double> extractVector(List<dynamic> raw) {
      List<double> v = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        v.add(raw[i] as double);
      }
      return v;
    }

    wq.pushData(extractFlatMatrix(newWeights['Wq']!));
    bq.pushData(extractVector(newWeights['bq']!));
    wk.pushData(extractFlatMatrix(newWeights['Wk']!));
    bk.pushData(extractVector(newWeights['bk']!));
    wv.pushData(extractFlatMatrix(newWeights['Wv']!));
    bv.pushData(extractVector(newWeights['bv']!));
    wo.pushData(extractFlatMatrix(newWeights['Wo']!));
    bo.pushData(extractVector(newWeights['bo']!));
  }
}
