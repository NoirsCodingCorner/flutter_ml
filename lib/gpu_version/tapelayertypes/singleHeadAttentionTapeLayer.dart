import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Applies Scaled Dot-Product Attention over a sequence.
/// Projects the input into Query, Key, and Value matrices to determine contextual relationships
/// between tokens before re-combining them based on the calculated attention scores.
class SingleHeadAttentionTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'SingleHeadAttentionTapeLayer';

  int dModel;
  int dK;
  int dV;

  /// Learnable weight matrix to project the input into Queries.
  late GPUTensor<Matrix> Wq;
  /// Learnable weight matrix to project the input into Keys.
  late GPUTensor<Matrix> Wk;
  /// Learnable weight matrix to project the input into Values.
  late GPUTensor<Matrix> Wv;

  /// A reference to the attention probabilities from the most recent forward pass.
  GPUTensor<Matrix>? lastAttentionWeights;

  /// --- Persistent Cache for Static Unrolling ---
  int cacheSeqLength = -1;
  List<GPUTensor<dynamic>> stepCache = <GPUTensor<dynamic>>[];

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
      params.add(Wq);
      params.add(Wk);
      params.add(Wv);
    }
    return params;
  }

  /// Allocates VRAM for the Query, Key, and Value projection matrices.
  @override
  void build(GPUTensor<Matrix> input) {
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

  /// Appends the single-head attention operations to the [tape].
  /// Persistently caches all intermediates to guarantee static unroll-ability.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    bool useCache = (cacheSeqLength == seqLength);

    if (!useCache) {
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = seqLength;
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

    GPUTensor<Matrix> q = matMulGPU(input, Wq, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(q);

    GPUTensor<Matrix> k = matMulGPU(input, Wk, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(k);

    GPUTensor<Matrix> v = matMulGPU(input, Wv, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(v);

    GPUTensor<Matrix> kT = transposeGPU(k, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(kT);

    GPUTensor<Matrix> scores = matMulGPU(q, kT, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(scores);

    double scaleFactor = 1.0 / sqrt(dK);
    GPUTensor<Matrix> scaledScores = scaleMatrixGPU(scores, scaleFactor, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(scaledScores);

    GPUTensor<Matrix> attentionWeights = softmaxMatrixGPU(scaledScores, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(attentionWeights);
    lastAttentionWeights = attentionWeights;

    GPUTensor<Matrix> output = matMulGPU(attentionWeights, v, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(output);

    return output;
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
      Wq.free();
      Wk.free();
      Wv.free();

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