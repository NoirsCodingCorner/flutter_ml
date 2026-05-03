import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class MultiHeadAttentionTL extends TapeLayer {
  int dModel;
  int numHeads;
  late int dHead;

  // The 8 Monolithic Parameters for BERT
  late GPUTensor<Matrix> Wq;
  late GPUTensor<Vector> bq;
  late GPUTensor<Matrix> Wk;
  late GPUTensor<Vector> bk;
  late GPUTensor<Matrix> Wv;
  late GPUTensor<Vector> bv;
  late GPUTensor<Matrix> Wo;
  late GPUTensor<Vector> bo;
  GPUTensor<Vector>? attentionMask; // Add this!

  MultiHeadAttentionTL(this.dModel, this.numHeads) {
    dHead = dModel ~/ numHeads;
  }

  @override
  String get name {
    return 'MultiHeadAttentionTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(Wq);
      params.add(bq);
      params.add(Wk);
      params.add(bk);
      params.add(Wv);
      params.add(bv);
      params.add(Wo);
      params.add(bo);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    // ⚡ Instant GPU-side VRAM allocation & Randomization!
    double scale = sqrt(2.0 / (dModel + dModel));

    Wq = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale);
    bq = GPUTensor<Vector>.empty(<int>[dModel]);

    Wk = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale);
    bk = GPUTensor<Vector>.empty(<int>[dModel]);

    Wv = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale);
    bv = GPUTensor<Vector>.empty(<int>[dModel]);

    Wo = GPUTensor<Matrix>.randomUniform(<int>[dModel, dModel], scale);
    bo = GPUTensor<Vector>.empty(<int>[dModel]);

    built = true;
  }

  // Magic Mapping Function
  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      map[prefix + '.self.query.weight'] = Wq;
      map[prefix + '.self.query.bias'] = bq;

      map[prefix + '.self.key.weight'] = Wk;
      map[prefix + '.self.key.bias'] = bk;

      map[prefix + '.self.value.weight'] = Wv;
      map[prefix + '.self.value.bias'] = bv;

      map[prefix + '.output.dense.weight'] = Wo;
      map[prefix + '.output.dense.bias'] = bo;
    }
    return map;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    // 1. Compute Full Q, K, V (Huge performance boost over per-head MatMuls)
    GPUTensor<Matrix> qMatMul = matMulGPU(typedInput, Wq, tape);
    intermediates.add(qMatMul);
    GPUTensor<Matrix> Q = addBiasToMatMulOutGPU(qMatMul, bq, tape);
    intermediates.add(Q);

    GPUTensor<Matrix> kMatMul = matMulGPU(typedInput, Wk, tape);
    intermediates.add(kMatMul);
    GPUTensor<Matrix> K = addBiasToMatMulOutGPU(kMatMul, bk, tape);
    intermediates.add(K);

    GPUTensor<Matrix> vMatMul = matMulGPU(typedInput, Wv, tape);
    intermediates.add(vMatMul);
    GPUTensor<Matrix> V = addBiasToMatMulOutGPU(vMatMul, bv, tape);
    intermediates.add(V);

    List<GPUTensor<Matrix>> headOutputs = <GPUTensor<Matrix>>[];
    double scaleFactor = 1.0 / sqrt(dHead);

    // 2. Slice and process each head independently
    for (int i = 0; i < numHeads; i = i + 1) {
      int startCol = i * dHead;
      int endCol = startCol + dHead;

      GPUTensor<Matrix> qHead = sliceColumnGPU(Q, startCol, endCol, tape);
      intermediates.add(qHead);

      GPUTensor<Matrix> kHead = sliceColumnGPU(K, startCol, endCol, tape);
      intermediates.add(kHead);

      GPUTensor<Matrix> vHead = sliceColumnGPU(V, startCol, endCol, tape);
      intermediates.add(vHead);

      // Attention Equation: Softmax(Q * K^T / sqrt(dHead)) * V
      GPUTensor<Matrix> kHeadT = transposeGPU(kHead, tape);
      intermediates.add(kHeadT);

      GPUTensor<Matrix> scores = matMulGPU(qHead, kHeadT, tape);
      intermediates.add(scores);

      GPUTensor<Matrix> scaledScores = scaleMatrixGPU(scores, scaleFactor, tape);
      intermediates.add(scaledScores);

      // ⚡ APPLY THE MASK BEFORE SOFTMAX
      GPUTensor<Matrix> maskedScores;
      if (attentionMask != null) {
        maskedScores = broadcastAddVectorToMatrixGPU(scaledScores, attentionMask!, tape);
        intermediates.add(maskedScores);
      } else {
        maskedScores = scaledScores;
      }

      GPUTensor<Matrix> probs = softmaxMatrixGPU(maskedScores, tape);
      intermediates.add(probs);

      GPUTensor<Matrix> headOut = matMulGPU(probs, vHead, tape);
      intermediates.add(headOut);

      headOutputs.add(headOut);
    }

    // 3. Concatenate all heads horizontally
    GPUTensor<Matrix> concatOut = concatenateMatricesByColumnGPU(headOutputs, tape);
    intermediates.add(concatOut);

    // 4. Final Output Projection
    GPUTensor<Matrix> oMatMul = matMulGPU(concatOut, Wo, tape);
    intermediates.add(oMatMul);
    GPUTensor<Matrix> finalOut = addBiasToMatMulOutGPU(oMatMul, bo, tape);
    intermediates.add(finalOut);

    return finalOut;
  }

  @override
  void free() {
    if (built) {
      Wq.free(); bq.free();
      Wk.free(); bk.free();
      Wv.free(); bv.free();
      Wo.free(); bo.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}