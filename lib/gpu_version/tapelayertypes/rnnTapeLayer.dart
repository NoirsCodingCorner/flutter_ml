import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// A standard Recurrent Neural Network (RNN) Layer.
/// Processes a sequence of data sequentially, maintaining a hidden state that is updated at each time step
/// based on the current input and the previous hidden state.
class RNNTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'RNNTapeLayer';


  int hiddenSize;
  String activation;

  /// Learnable weight matrix transforming the current input.
  late GPUTensor<Matrix> W_xh;
  /// Learnable weight matrix transforming the previous hidden state.
  late GPUTensor<Matrix> W_hh;
  /// Learnable bias column matrix added to the combined input and hidden state.
  late GPUTensor<Matrix> b_h;

  /// Persistent Cache for Static Unrolling
  int cacheSeqLength = -1;
  List<GPUTensor<Matrix>> hStates = <GPUTensor<Matrix>>[];
  List<GPUTensor<dynamic>> stepCache = <GPUTensor<dynamic>>[];

  /// Requires the size of the hidden state [hiddenSize].
  /// The [activation] can be either 'relu' or 'tanh'.
  RNNTL(this.hiddenSize, {this.activation = 'relu'});

  /// Returns the trainable weight matrices and bias.
  /// Specifically [W_xh],[W_hh] and [b_h].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(W_xh);
      params.add(W_hh);
      params.add(b_h);
    }
    return params;
  }

  /// Allocates VRAM for weights and biases using a standard Xavier initialization.
  @override
  void build(GPUTensor<Matrix> input) {
    int inputSize = input.shape[1];

    Random random = Random();
    double xavierStdDev(int fanIn, int fanOut) {
      return sqrt(2.0 / (fanIn + fanOut));
    }

    double inputToHiddenStdDev = xavierStdDev(inputSize, hiddenSize);
    List<List<double>> wXhValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < inputSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * inputToHiddenStdDev);
      }
      wXhValues.add(row);
    }

    double hiddenToHiddenStdDev = xavierStdDev(hiddenSize, hiddenSize);
    List<List<double>> wHhValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < hiddenSize; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * hiddenToHiddenStdDev);
      }
      wHhValues.add(row);
    }

    W_xh = GPUTensor<Matrix>(wXhValues);
    W_hh = GPUTensor<Matrix>(wHhValues);

    List<List<double>> bHValues = <List<double>>[];
    for (int i = 0; i < hiddenSize; i = i + 1) {
      bHValues.add(<double>[0.0]);
    }
    b_h = GPUTensor<Matrix>(bHValues);

    built = true;
  }

  /// Appends the recurrent step-by-step operations to the [tape].
  /// Persistently caches all intermediates to guarantee static unrollability.
  /// The [intermediates] list should be empty since it is not used in this operation.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    bool useCache = (cacheSeqLength == seqLength);

    if (!useCache) {
      for (int i = 0; i < hStates.length; i = i + 1) { hStates[i].free(); }
      for (int i = 0; i < stepCache.length; i = i + 1) { stepCache[i].free(); }

      hStates.clear();
      stepCache.clear();
      cacheSeqLength = seqLength;

      List<List<double>> initialHValues = <List<double>>[];
      for (int i = 0; i < hiddenSize; i = i + 1) {
        initialHValues.add(<double>[0.0]);
      }

      hStates.add(GPUTensor<Matrix>(initialHValues));
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

    GPUTensor<Matrix> transposedInput = transposeGPU(input, tape, outTensor: getCached<GPUTensor<Matrix>>());
    saveCached(transposedInput);

    for (int i = 0; i < seqLength; i = i + 1) {
      GPUTensor<Matrix> hPrev = hStates[i];

      // Extract time step as a strict [InputSize, 1] Matrix
      GPUTensor<Matrix> xT = sliceColumnGPU(transposedInput, i, i + 1, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(xT);

      GPUTensor<Matrix> inputPart = matMulGPU(W_xh, xT, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(inputPart);

      GPUTensor<Matrix> hiddenPart = matMulGPU(W_hh, hPrev, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(hiddenPart);

      GPUTensor<Matrix> sum1 = addMatrixGPU(inputPart, hiddenPart, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(sum1);

      GPUTensor<Matrix> combined = addMatrixGPU(sum1, b_h, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(combined);

      GPUTensor<Matrix>? outNewH = useCache ? hStates[i + 1] : null;
      GPUTensor<Matrix> newH;

      if (activation == 'relu') {
        newH = reluMatrixGPU(combined, tape, outTensor: outNewH);
      } else {
        newH = tanhMatrixGPU(combined, tape, outTensor: outNewH);
      }

      if (!useCache) {
        hStates.add(newH);
      }
    }

    // Return the pure 2D Matrix representation of the final hidden state
    return hStates.last;
  }

  /// Clears the gradients of all persistently cached intermediate tensors across all timesteps.
  @override
  void zeroStates(CommandBuffer tape) {
    for (int i = 0; i < hStates.length; i = i + 1) {
      hStates[i].zeroGrad(tape);
    }
    for (int i = 0; i < stepCache.length; i = i + 1) {
      stepCache[i].zeroGrad(tape);
    }
  }

  /// Frees VRAM for all parameters and persistently cached intermediate tensors.
  @override
  void free() {
    if (built) {
      W_xh.free();
      W_hh.free();
      b_h.free();

      for (int i = 0; i < hStates.length; i = i + 1) { hStates[i].free(); }
      for (int i = 0; i < stepCache.length; i = i + 1) { stepCache[i].free(); }

      hStates.clear();
      stepCache.clear();
      cacheSeqLength = -1;
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    W_xh.toCpu();
    W_hh.toCpu();
    b_h.toCpu();

    wMap['W_xh'] = W_xh.value;
    wMap['W_hh'] = W_hh.value;
    wMap['b_h'] = b_h.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      free();
    }

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

    W_xh = GPUTensor<Matrix>(extractMatrix(newWeights['W_xh']!));
    W_hh = GPUTensor<Matrix>(extractMatrix(newWeights['W_hh']!));
    b_h = GPUTensor<Matrix>(extractMatrix(newWeights['b_h']!));

    built = true;
  }
}