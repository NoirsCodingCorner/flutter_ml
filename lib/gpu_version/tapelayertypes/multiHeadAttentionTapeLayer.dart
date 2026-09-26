import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

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
  late GPUTensor<Matrix> Wq;
  late GPUTensor<Vector> bq;
  late GPUTensor<Matrix> Wk;
  late GPUTensor<Vector> bk;
  late GPUTensor<Matrix> Wv;
  late GPUTensor<Vector> bv;
  late GPUTensor<Matrix> Wo;
  late GPUTensor<Vector> bo;

  /// Optional mask added to the attention scores before the softmax operation.
  GPUTensor<Vector>? attentionMask;

  /// Persistent Cache for Static Unrolling
  int cacheSeqLength = -1;
  bool cacheUsedMask = false;
  final List<GPUTensor<dynamic>> stepCache = <GPUTensor<dynamic>>[];

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
      params.addAll(<GPUTensor>[Wq, bq, Wk, bk, Wv, bv, Wo, bo]);
    }
    return params;
  }

  /// Allocates VRAM for the Query, Key, Value, and Output projections.
  /// Uses a scaled random uniform distribution for initial weights and zeros for biases.
  @override
  void build(GPUTensor<Matrix> input) {
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

  /// Used specifically for SafeTensors compatibility to map weights to standard transformer nomenclature.
  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      map['$prefix.self.query.weight'] = Wq;
      map['$prefix.self.query.bias'] = bq;

      map['$prefix.self.key.weight'] = Wk;
      map['$prefix.self.key.bias'] = bk;

      map['$prefix.self.value.weight'] = Wv;
      map['$prefix.self.value.bias'] = bv;

      map['$prefix.output.dense.weight'] = Wo;
      map['$prefix.output.dense.bias'] = bo;
    }
    return map;
  }

  /// Appends the multi-head attention operations to the [tape].
  /// Persistently caches all intermediates to guarantee static unroll-ability.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    bool hasMask = (attentionMask != null);

    // The cache is only valid if sequence length AND masking state are identical to the previous pass
    bool useCache = (cacheSeqLength == seqLength && cacheUsedMask == hasMask);

    if (!useCache) {
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = seqLength;
      cacheUsedMask = hasMask;
    }

    int cIdx = 0;

    T? getCached<T>() {
      if (useCache) {
        T cached = stepCache[cIdx] as T;
        cIdx = cIdx + 1;
        return cached;
      }
      return null;
    }

    void saveCached(GPUTensor<dynamic> tensor) {
      if (!useCache) {
        stepCache.add(tensor);
      }
    }

    // 1. Compute Full Q, K, V
    GPUTensor<Matrix> qMatMul = matMulGPU(input, Wq, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(qMatMul);
    GPUTensor<Matrix> Q = addBiasToMatMulOutGPU(qMatMul, bq, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(Q);

    GPUTensor<Matrix> kMatMul = matMulGPU(input, Wk, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(kMatMul);
    GPUTensor<Matrix> K = addBiasToMatMulOutGPU(kMatMul, bk, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(K);

    GPUTensor<Matrix> vMatMul = matMulGPU(input, Wv, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(vMatMul);
    GPUTensor<Matrix> V = addBiasToMatMulOutGPU(vMatMul, bv, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(V);

    List<GPUTensor<Matrix>> headOutputs = <GPUTensor<Matrix>>[];
    double scaleFactor = 1.0 / sqrt(dHead);

    // 2. Slice and process each head independently
    for (int i = 0; i < numHeads; i = i + 1) {
      int startCol = i * dHead;
      int endCol = startCol + dHead;

      GPUTensor<Matrix> qHead = sliceColumnGPU(Q, startCol, endCol, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(qHead);
      GPUTensor<Matrix> kHead = sliceColumnGPU(K, startCol, endCol, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(kHead);
      GPUTensor<Matrix> vHead = sliceColumnGPU(V, startCol, endCol, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(vHead);

      // Attention Equation: Softmax(Q * K^T / sqrt(dHead)) * V
      GPUTensor<Matrix> kHeadT = transposeGPU(kHead, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(kHeadT);
      GPUTensor<Matrix> scores = matMulGPU(qHead, kHeadT, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(scores);
      GPUTensor<Matrix> scaledScores = scaleMatrixGPU(scores, scaleFactor, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(scaledScores);

      GPUTensor<Matrix> maskedScores;
      if (hasMask) {
        maskedScores = broadcastAddVectorToMatrixGPU(scaledScores, attentionMask!, tape, outTensor: getCached<GPUTensor<Matrix>>());
        saveCached(maskedScores);
      } else {
        maskedScores = scaledScores;
      }

      GPUTensor<Matrix> probs = softmaxMatrixGPU(maskedScores, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(probs);
      GPUTensor<Matrix> headOut = matMulGPU(probs, vHead, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(headOut);

      headOutputs.add(headOut);
    }

    // 3. Concatenate all heads horizontally
    GPUTensor<Matrix> concatOut = concatenateMatricesByColumnGPU(headOutputs, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(concatOut);

    // 4. Final Output Projection
    GPUTensor<Matrix> oMatMul = matMulGPU(concatOut, Wo, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(oMatMul);
    GPUTensor<Matrix> finalOut = addBiasToMatMulOutGPU(oMatMul, bo, tape, outTensor: getCached<GPUTensor<Matrix>>()); saveCached(finalOut);

    return finalOut;
  }

  /// Clears the gradients of all persistently cached intermediate tensors.
  @override
  void zeroStates(CommandBuffer tape) {
    for (int i = 0; i < stepCache.length; i = i + 1) {
      stepCache[i].zeroGrad(tape);
    }
  }

  /// Frees VRAM for all parameters and persistently cached intermediate tensors.
  @override
  void free() {
    if (built) {
      Wq.free(); bq.free();
      Wk.free(); bk.free();
      Wv.free(); bv.free();
      Wo.free(); bo.free();

      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = -1;
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    Wq.toCpu(); bq.toCpu();
    Wk.toCpu(); bk.toCpu();
    Wv.toCpu(); bv.toCpu();
    Wo.toCpu(); bo.toCpu();

    wMap['Wq'] = Wq.value; wMap['bq'] = bq.value;
    wMap['Wk'] = Wk.value; wMap['bk'] = bk.value;
    wMap['Wv'] = Wv.value; wMap['bv'] = bv.value;
    wMap['Wo'] = Wo.value; wMap['bo'] = bo.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) free();

    List<List<double>> extractMatrix(List<dynamic> raw) {
      List<List<double>> m = <List<double>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<double> row = <double>[];
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          row.add(rawRow[j] as double);
        }
        m.add(row);
      }
      return m;
    }

    List<double> extractVector(List<dynamic> raw) {
      List<double> v = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        v.add(raw[i] as double);
      }
      return v;
    }

    Wq = GPUTensor<Matrix>(extractMatrix(newWeights['Wq']!));
    bq = GPUTensor<Vector>(extractVector(newWeights['bq']!));
    Wk = GPUTensor<Matrix>(extractMatrix(newWeights['Wk']!));
    bk = GPUTensor<Vector>(extractVector(newWeights['bk']!));
    Wv = GPUTensor<Matrix>(extractMatrix(newWeights['Wv']!));
    bv = GPUTensor<Vector>(extractVector(newWeights['bv']!));
    Wo = GPUTensor<Matrix>(extractMatrix(newWeights['Wo']!));
    bo = GPUTensor<Vector>(extractVector(newWeights['bo']!));

    built = true;
  }
}