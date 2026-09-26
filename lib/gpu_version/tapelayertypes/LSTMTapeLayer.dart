import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Standard Long Short-Term Memory (LSTM) Layer.
/// Processes a sequence of data to capture long-term dependencies.
class LSTMTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'LSTMTapeLayer';

  int hiddenSize;
  double gradClipValue;

  late GPUTensor<Matrix> W_xf;
  late GPUTensor<Matrix> W_hf;
  late GPUTensor<Matrix> b_f;
  late GPUTensor<Matrix> W_xi;
  late GPUTensor<Matrix> W_hi;
  late GPUTensor<Matrix> b_i;
  late GPUTensor<Matrix> W_xc;
  late GPUTensor<Matrix> W_hc;
  late GPUTensor<Matrix> b_c;
  late GPUTensor<Matrix> W_xo;
  late GPUTensor<Matrix> W_ho;
  late GPUTensor<Matrix> b_o;

  /// Persistent Cache for Static Unrolling
  int cacheSeqLength = -1;
  final List<GPUTensor<Matrix>> hStates = <GPUTensor<Matrix>>[];
  final List<GPUTensor<Matrix>> cStates = <GPUTensor<Matrix>>[];
  final List<GPUTensor<dynamic>> stepCache = <GPUTensor<dynamic>>[];

  /// Requires the number of hidden features in both the cell state and the hidden state [hiddenSize].
  /// Optionally a [gradClipValue] can be assigned to prevent gradient explosion.
  LSTMTL(this.hiddenSize, {this.gradClipValue = 1.0});

  /// Returns all trainable parameters of the layer.
  /// Specifically returns [W_xf],[W_hf],[b_f],[W_xi],[W_hi],[b_i],[W_xc],[W_hc],[b_c],[W_xo],[W_ho] and [b_o].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(<GPUTensor>[
        W_xf,
        W_hf,
        b_f,
        W_xi,
        W_hi,
        b_i,
        W_xc,
        W_hc,
        b_c,
        W_xo,
        W_ho,
        b_o,
      ]);
    }
    return params;
  }

  /// Allocates VRAM for all gates and initializes them with random values.
  @override
  void build(GPUTensor<Matrix> input) {
    int inputSize = input.shape[1];
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
      for (int i = 0; i < hiddenSize; i = i + 1) {
        row.add(0.0);
      }
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

  // Helper to strictly slice a sequence row into a 1xN Matrix.
  // Added outTensor support for static unrolling caching.
  GPUTensor<Matrix> sliceRowToMatrix(
      GPUTensor<Matrix> sequence,
      int rowIdx,
      CommandBuffer tape, {
        GPUTensor<Matrix>? outTensor,
      }) {
    GPUTensor<Matrix> out =
        outTensor ?? GPUTensor<Matrix>.empty(<int>[1, sequence.shape[1]]);
    tape.putInt(OP_SLICE_ROW);
    tape.putString(sequence.id);
    tape.putString(out.id);
    tape.putInt(rowIdx);

    if (outTensor == null) {
      out.creator = GPUNode(<GPUTensor>[sequence], (CommandBuffer bTape) {
        bTape.putInt(OP_SLICE_ROW_BACKWARD);
        bTape.putString('${out.id}_grad');
        bTape.putString('${sequence.id}_grad');
        bTape.putInt(rowIdx);
      }, opName: 'slice_row_matrix_gpu');
    }
    return out;
  }

  /// Appends the recurrent sequential operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  /// To support static tape unrolling, this layer now persistently caches its own intermediates
  /// and bypasses the external [intermediates] list. It dynamically reallocates memory only if the sequence length changes.
  @override
  GPUTensor<Matrix> forward(
      GPUTensor<Matrix> input,
      CommandBuffer tape,
      List<GPUTensor> intermediates,
      ) {
    int seqLength = input.shape[0];
    bool useCache = (cacheSeqLength == seqLength);

    if (!useCache) {
      for (var t in hStates) {
        t.free();
      }
      for (var t in cStates) {
        t.free();
      }
      for (var t in stepCache) {
        t.free();
      }

      hStates.clear();
      cStates.clear();
      stepCache.clear();
      cacheSeqLength = seqLength;

      List<List<double>> zeroMatrix = <List<double>>[];
      List<double> zeroRow = <double>[];
      for (int i = 0; i < hiddenSize; i = i + 1) {
        zeroRow.add(0.0);
      }
      zeroMatrix.add(zeroRow);

      hStates.add(GPUTensor<Matrix>(zeroMatrix));
      cStates.add(GPUTensor<Matrix>(zeroMatrix));
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

    for (int t = 0; t < seqLength; t = t + 1) {
      GPUTensor<Matrix> xT = sliceRowToMatrix(
        input,
        t,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(xT);

      GPUTensor<Matrix> prevH = hStates[t];
      GPUTensor<Matrix> prevC = cStates[t];

      // Forget Gate
      GPUTensor<Matrix> fTX = matMulGPU(
        xT,
        W_xf,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(fTX);
      GPUTensor<Matrix> fTH = matMulGPU(
        prevH,
        W_hf,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(fTH);
      GPUTensor<Matrix> fTSum = addMatrixGPU(
        fTX,
        fTH,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(fTSum);
      GPUTensor<Matrix> fTBiased = addMatrixGPU(
        fTSum,
        b_f,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(fTBiased);
      GPUTensor<Matrix> fT = sigmoidMatrixGPU(
        fTBiased,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(fT);

      // Input Gate
      GPUTensor<Matrix> iTX = matMulGPU(
        xT,
        W_xi,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(iTX);
      GPUTensor<Matrix> iTH = matMulGPU(
        prevH,
        W_hi,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(iTH);
      GPUTensor<Matrix> iTSum = addMatrixGPU(
        iTX,
        iTH,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(iTSum);
      GPUTensor<Matrix> iTBiased = addMatrixGPU(
        iTSum,
        b_i,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(iTBiased);
      GPUTensor<Matrix> iT = sigmoidMatrixGPU(
        iTBiased,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(iT);

      // Cell Candidate
      GPUTensor<Matrix> cTildeTX = matMulGPU(
        xT,
        W_xc,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cTildeTX);
      GPUTensor<Matrix> cTildeTH = matMulGPU(
        prevH,
        W_hc,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cTildeTH);
      GPUTensor<Matrix> cTildeTSum = addMatrixGPU(
        cTildeTX,
        cTildeTH,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cTildeTSum);
      GPUTensor<Matrix> cTildeTBiased = addMatrixGPU(
        cTildeTSum,
        b_c,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cTildeTBiased);
      GPUTensor<Matrix> cTildeT = tanhMatrixGPU(
        cTildeTBiased,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cTildeT);

      // Cell State Update
      GPUTensor<Matrix> cRetained = elementWiseMultiplyMatrixGPU(
        fT,
        prevC,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cRetained);
      GPUTensor<Matrix> cNewInfo = elementWiseMultiplyMatrixGPU(
        iT,
        cTildeT,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cNewInfo);

      GPUTensor<Matrix>? outNewC = useCache ? cStates[t + 1] : null;
      GPUTensor<Matrix> newC = addMatrixGPU(
        cRetained,
        cNewInfo,
        tape,
        outTensor: outNewC,
      );
      if (!useCache) cStates.add(newC);

      // Output Gate
      GPUTensor<Matrix> oTX = matMulGPU(
        xT,
        W_xo,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(oTX);
      GPUTensor<Matrix> oTH = matMulGPU(
        prevH,
        W_ho,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(oTH);
      GPUTensor<Matrix> oTSum = addMatrixGPU(
        oTX,
        oTH,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(oTSum);
      GPUTensor<Matrix> oTBiased = addMatrixGPU(
        oTSum,
        b_o,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(oTBiased);
      GPUTensor<Matrix> oT = sigmoidMatrixGPU(
        oTBiased,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(oT);

      // Hidden State Update
      GPUTensor<Matrix> cActivated = tanhMatrixGPU(
        newC,
        tape,
        outTensor: getCached<GPUTensor<Matrix>>(),
      );
      saveCached(cActivated);

      GPUTensor<Matrix>? outNewH = useCache ? hStates[t + 1] : null;
      GPUTensor<Matrix> newH = elementWiseMultiplyMatrixGPU(
        oT,
        cActivated,
        tape,
        outTensor: outNewH,
      );
      if (!useCache) hStates.add(newH);
    }

    GPUTensor<Matrix> finalH = hStates.last;

    // Inject Gradient Clipping
    GPUNode? originalCreator = finalH.creator;
    if (originalCreator != null) {
      void Function(CommandBuffer) originalBackward =
          originalCreator.backwardFn;
      originalCreator.backwardFn = (CommandBuffer bTape) {
        originalBackward(bTape);
        List<GPUTensor> params = parameters;
        for (int p = 0; p < params.length; p = p + 1) {
          bTape.putInt(OP_CLIP_GRAD_VALUE);
          bTape.putString('${params[p].id}_grad');
          bTape.putFloat(gradClipValue);
        }
      };
    }

    return finalH;
  }

  /// Clears the gradients of all persistently cached intermediate tensors across all timesteps.
  @override
  void zeroStates(CommandBuffer tape) {
    for (int i = 0; i < hStates.length; i = i + 1) {
      hStates[i].zeroGrad(tape);
    }
    for (int i = 0; i < cStates.length; i = i + 1) {
      cStates[i].zeroGrad(tape);
    }
    for (int i = 0; i < stepCache.length; i = i + 1) {
      stepCache[i].zeroGrad(tape);
    }
  }

  /// Frees VRAM for all allocated weights, biases, and persistently cached intermediates.
  @override
  void free() {
    if (built) {
      W_xf.free();
      W_hf.free();
      b_f.free();
      W_xi.free();
      W_hi.free();
      b_i.free();
      W_xc.free();
      W_hc.free();
      b_c.free();
      W_xo.free();
      W_ho.free();
      b_o.free();

      for (var t in hStates) {
        t.free();
      }
      for (var t in cStates) {
        t.free();
      }
      for (var t in stepCache) {
        t.free();
      }

      hStates.clear();
      cStates.clear();
      stepCache.clear();
      cacheSeqLength = -1;
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    W_xf.toCpu();
    W_hf.toCpu();
    b_f.toCpu();
    W_xi.toCpu();
    W_hi.toCpu();
    b_i.toCpu();
    W_xc.toCpu();
    W_hc.toCpu();
    b_c.toCpu();
    W_xo.toCpu();
    W_ho.toCpu();
    b_o.toCpu();

    wMap['W_xf'] = W_xf.value;
    wMap['W_hf'] = W_hf.value;
    wMap['b_f'] = b_f.value;
    wMap['W_xi'] = W_xi.value;
    wMap['W_hi'] = W_hi.value;
    wMap['b_i'] = b_i.value;
    wMap['W_xc'] = W_xc.value;
    wMap['W_hc'] = W_hc.value;
    wMap['b_c'] = b_c.value;
    wMap['W_xo'] = W_xo.value;
    wMap['W_ho'] = W_ho.value;
    wMap['b_o'] = b_o.value;

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

    W_xf = GPUTensor<Matrix>(extractMatrix(newWeights['W_xf']!));
    W_hf = GPUTensor<Matrix>(extractMatrix(newWeights['W_hf']!));
    b_f = GPUTensor<Matrix>(extractMatrix(newWeights['b_f']!));

    W_xi = GPUTensor<Matrix>(extractMatrix(newWeights['W_xi']!));
    W_hi = GPUTensor<Matrix>(extractMatrix(newWeights['W_hi']!));
    b_i = GPUTensor<Matrix>(extractMatrix(newWeights['b_i']!));

    W_xc = GPUTensor<Matrix>(extractMatrix(newWeights['W_xc']!));
    W_hc = GPUTensor<Matrix>(extractMatrix(newWeights['W_hc']!));
    b_c = GPUTensor<Matrix>(extractMatrix(newWeights['b_c']!));

    W_xo = GPUTensor<Matrix>(extractMatrix(newWeights['W_xo']!));
    W_ho = GPUTensor<Matrix>(extractMatrix(newWeights['W_ho']!));
    b_o = GPUTensor<Matrix>(extractMatrix(newWeights['b_o']!));

    built = true;
  }
}