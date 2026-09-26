import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Advanced Layer to process sequences of 2D spatial data. In this case it is represented as
/// Tensor3Ds with format [Time, Height, Width]. It uses convolutional gates instead of standard matrix multiplications.
/// Therefore this architecture is able to capture both time and spatial information.
class ConvLSTMTL extends TapeLayer<Tensor3D,Tensor3D> {
  String get name => 'ConvLSTMTapeLayer';

  int hiddenFilters;
  int kernelSize;
  double gradClipValue;

  // Forget gate
  /// Learnable input convolution kernel for the forget gate.
  late GPUTensor<Tensor3D> K_xf;
  /// Learnable hidden state convolution kernel for the forget gate.
  late GPUTensor<Tensor3D> K_hf;

  // Input gate
  /// Learnable input convolution kernel for the input gate.
  late GPUTensor<Tensor3D> K_xi;
  /// Learnable hidden state convolution kernel for the input gate.
  late GPUTensor<Tensor3D> K_hi;

  // Cell candidate
  /// Learnable input convolution kernel for the cell candidate.
  late GPUTensor<Tensor3D> K_xc;
  /// Learnable hidden state convolution kernel for the cell candidate.
  late GPUTensor<Tensor3D> K_hc;

  // Output gate
  /// Learnable input convolution kernel for the output gate.
  late GPUTensor<Tensor3D> K_xo;
  /// Learnable hidden state convolution kernel for the output gate.
  late GPUTensor<Tensor3D> K_ho;


  /// Learnable bias vector for the forget gate.
  late GPUTensor<Vector> b_f;
  /// Learnable bias vector for the input gate.
  late GPUTensor<Vector> b_i;
  /// Learnable bias vector for the cell candidate.
  late GPUTensor<Vector> b_c;
  /// Learnable bias vector for the output gate.
  late GPUTensor<Vector> b_o;

  /// Non-trainable zeroBias to skip redundant bias additions in hidden state convolutions.
  late GPUTensor<Vector> zeroBiasInput;

  /// --- Persistent Cache for Static Unrolling ---
  int cacheSeqLength = -1;
  final List<GPUTensor<Tensor3D>> hStates = <GPUTensor<Tensor3D>>[];
  final List<GPUTensor<Tensor3D>> cStates = <GPUTensor<Tensor3D>>[];
  final List<GPUTensor<dynamic>> stepCache = <GPUTensor<dynamic>>[];


  /// Requires the number of desired [hiddenFilters] and the size of the kernels [kernelSize]x[kernelSize].
  /// Additionally to help with exploding gradients [gradClipValue] clips gradients during backpropagation.
  ConvLSTMTL(this.hiddenFilters, this.kernelSize, {this.gradClipValue = 1.0});


  /// Returns all trainable kernels and biases for the LSTM gates.
  /// Specifically returns [K_xf],[K_hf],[b_f],[K_xi],[K_hi],[b_i],[K_xc],[K_hc],[b_c],[K_xo],[K_ho] and [b_o].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(K_xf);
      params.add(K_hf);
      params.add(b_f);
      params.add(K_xi);
      params.add(K_hi);
      params.add(b_i);
      params.add(K_xc);
      params.add(K_hc);
      params.add(b_c);
      params.add(K_xo);
      params.add(K_ho);
      params.add(b_o);
    }
    return params;
  }

  /// Allocates VRAM for all kernels and biases. Weights are initialized using scaled random values.
  @override
  void build(GPUTensor<dynamic> input) {
    GPUTensor<Tensor3D> inputSequence = input as GPUTensor<Tensor3D>;
    int inChannels = 1; // The sequence is [Time, Height, Width] so we process 1 channel slices
    Random random = Random();

    List<List<List<double>>> initWeight(int inC, int outC) {
      double scale = sqrt(2.0 / (inC + outC));
      List<List<List<double>>> weightData = <List<List<double>>>[];
      for (int oc = 0; oc < outC; oc = oc + 1) {
        List<List<double>> rows = <List<double>>[];
        for (int ic = 0; ic < inC; ic = ic + 1) {
          for (int h = 0; h < kernelSize; h = h + 1) {
            List<double> row = <double>[];
            for (int w = 0; w < kernelSize; w = w + 1) {
              row.add((random.nextDouble() * 2.0 - 1.0) * scale);
            }
            rows.add(row);
          }
        }
        weightData.add(rows);
      }
      return weightData;
    }

    List<double> initBias(int outC) {
      List<double> b = <double>[];
      for (int i = 0; i < outC; i = i + 1) {
        b.add(0.0);
      }
      return b;
    }

    K_xf = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    K_hf = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    K_xi = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    K_hi = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    K_xc = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    K_hc = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    K_xo = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    K_ho = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));

    b_f = GPUTensor<Vector>(initBias(hiddenFilters));
    b_i = GPUTensor<Vector>(initBias(hiddenFilters));
    b_c = GPUTensor<Vector>(initBias(hiddenFilters));
    b_o = GPUTensor<Vector>(initBias(hiddenFilters));

    zeroBiasInput = GPUTensor<Vector>(initBias(hiddenFilters));

    built = true;
  }

  /// Appends the recurrent sequential convolution operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  /// To support static tape unrolling, this layer now persistently caches its own intermediates
  /// and bypasses the external [intermediates] list. It dynamically reallocates memory only if the sequence length changes.
  @override
  GPUTensor<Tensor3D> forward(GPUTensor<Tensor3D> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];

    bool useCache = (cacheSeqLength == seqLength);

    // If sequence length changed or this is the first run, rebuild the cache.
    if (!useCache) {
      // Free old VRAM allocations to prevent memory leaks if sequence length changed dynamically
      for (int i = 0; i < hStates.length; i = i + 1) { hStates[i].free(); }
      for (int i = 0; i < cStates.length; i = i + 1) { cStates[i].free(); }
      for (int i = 0; i < stepCache.length; i = i + 1) { stepCache[i].free(); }

      hStates.clear();
      cStates.clear();
      stepCache.clear();
      cacheSeqLength = seqLength;

      List<List<List<double>>> zero3D = <List<List<double>>>[];
      for (int c = 0; c < hiddenFilters; c = c + 1) {
        List<List<double>> mat = <List<double>>[];
        for (int i = 0; i < height; i = i + 1) {
          List<double> row = <double>[];
          for (int j = 0; j < width; j = j + 1) {
            row.add(0.0);
          }
          mat.add(row);
        }
        zero3D.add(mat);
      }

      // Initialize state histories
      hStates.add(GPUTensor<Tensor3D>(zero3D));
      cStates.add(GPUTensor<Tensor3D>(zero3D));
    }

    int cIdx = 0;

    // Inline helpers to keep loop syntax perfectly clean
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
      GPUTensor<Matrix> xT = selectMatrixFrom3DGPU(input, t, tape, outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(xT);

      // Explicitly reference the previous timestep's states
      GPUTensor<Tensor3D> prevH = hStates[t];
      GPUTensor<Tensor3D> prevC = cStates[t];

      // Forget Gate
      GPUTensor<Tensor3D> fX = conv2dMultiChannelGPU(xT, K_xf, b_f, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fX);
      GPUTensor<Tensor3D> fH = conv2dMultiChannelGPU(prevH, K_hf, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fH);
      GPUTensor<Tensor3D> fSum = add3DGPU(fX, fH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fSum);
      GPUTensor<Tensor3D> fT = sigmoid3DGPU(fSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fT);

      // Input Gate
      GPUTensor<Tensor3D> iX = conv2dMultiChannelGPU(xT, K_xi, b_i, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iX);
      GPUTensor<Tensor3D> iH = conv2dMultiChannelGPU(prevH, K_hi, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iH);
      GPUTensor<Tensor3D> iSum = add3DGPU(iX, iH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iSum);
      GPUTensor<Tensor3D> iT = sigmoid3DGPU(iSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iT);

      // Cell Candidate
      GPUTensor<Tensor3D> cX = conv2dMultiChannelGPU(xT, K_xc, b_c, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cX);
      GPUTensor<Tensor3D> cH = conv2dMultiChannelGPU(prevH, K_hc, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cH);
      GPUTensor<Tensor3D> cSum = add3DGPU(cX, cH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cSum);
      GPUTensor<Tensor3D> cTildeT = tanh3DGPU(cSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cTildeT);

      // Cell State Update
      GPUTensor<Tensor3D> cRetained = elementWiseMultiply3DGPU(fT, prevC, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cRetained);
      GPUTensor<Tensor3D> cNewInfo = elementWiseMultiply3DGPU(iT, cTildeT, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cNewInfo);

      GPUTensor<Tensor3D>? outNewC = useCache ? cStates[t + 1] : null;
      GPUTensor<Tensor3D> newC = add3DGPU(cRetained, cNewInfo, tape, outTensor: outNewC);
      if (!useCache) cStates.add(newC);

      // Output Gate
      GPUTensor<Tensor3D> oX = conv2dMultiChannelGPU(xT, K_xo, b_o, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oX);
      GPUTensor<Tensor3D> oH = conv2dMultiChannelGPU(prevH, K_ho, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oH);
      GPUTensor<Tensor3D> oSum = add3DGPU(oX, oH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oSum);
      GPUTensor<Tensor3D> oT = sigmoid3DGPU(oSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oT);

      // Hidden State Update
      GPUTensor<Tensor3D> cActivated = tanh3DGPU(newC, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cActivated);

      GPUTensor<Tensor3D>? outNewH = useCache ? hStates[t + 1] : null;
      GPUTensor<Tensor3D> newH = elementWiseMultiply3DGPU(oT, cActivated, tape, outTensor: outNewH);
      if (!useCache) hStates.add(newH);
    }

    GPUTensor<Tensor3D> finalH = hStates.last;

    GPUNode? originalCreator = finalH.creator;
    if (originalCreator != null) {
      void Function(CommandBuffer) originalBackward = originalCreator.backwardFn;
      originalCreator.backwardFn = (CommandBuffer bTape) {
        List<GPUTensor> params = parameters;
        for (int i = 0; i < params.length; i = i + 1) {
          bTape.putInt(OP_CLIP_GRAD_VALUE);
          bTape.putString('${params[i].id}_grad');
          bTape.putFloat(gradClipValue);
        }
        originalBackward(bTape);
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

  /// Frees VRAM for all allocated kernels, biases, and persistently cached intermediates.
  @override
  void free() {
    if (built) {
      // Free parameters
      K_xf.free();
      K_hf.free();
      K_xi.free();
      K_hi.free();
      K_xc.free();
      K_hc.free();
      K_xo.free();
      K_ho.free();
      b_f.free();
      b_i.free();
      b_c.free();
      b_o.free();
      zeroBiasInput.free();

      // Free persistently cached states and intermediates
      for (int i = 0; i < hStates.length; i = i + 1) { hStates[i].free(); }
      for (int i = 0; i < cStates.length; i = i + 1) { cStates[i].free(); }
      for (int i = 0; i < stepCache.length; i = i + 1) { stepCache[i].free(); }

      hStates.clear();
      cStates.clear();
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

    K_xf.toCpu();
    K_hf.toCpu();
    b_f.toCpu();
    K_xi.toCpu();
    K_hi.toCpu();
    b_i.toCpu();
    K_xc.toCpu();
    K_hc.toCpu();
    b_c.toCpu();
    K_xo.toCpu();
    K_ho.toCpu();
    b_o.toCpu();

    wMap['K_xf'] = K_xf.value;
    wMap['K_hf'] = K_hf.value;
    wMap['b_f']  = b_f.value;
    wMap['K_xi'] = K_xi.value;
    wMap['K_hi'] = K_hi.value;
    wMap['b_i']  = b_i.value;
    wMap['K_xc'] = K_xc.value;
    wMap['K_hc'] = K_hc.value;
    wMap['b_c']  = b_c.value;
    wMap['K_xo'] = K_xo.value;
    wMap['K_ho'] = K_ho.value;
    wMap['b_o']  = b_o.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      free();
    }

    List<double> extractVector(List<dynamic> raw) {
      List<double> result = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        result.add(raw[i] as double);
      }
      return result;
    }

    List<List<List<double>>> extractTensor3D(List<dynamic> raw) {
      List<List<List<double>>> result = <List<List<double>>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<List<double>> channel = <List<double>>[];
        List<dynamic> rawChannel = raw[i] as List<dynamic>;
        for (int j = 0; j < rawChannel.length; j = j + 1) {
          List<double> row = <double>[];
          List<dynamic> rawRow = rawChannel[j] as List<dynamic>;
          for (int k = 0; k < rawRow.length; k = k + 1) {
            row.add(rawRow[k] as double);
          }
          channel.add(row);
        }
        result.add(channel);
      }
      return result;
    }

    K_xf = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xf']!));
    K_hf = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hf']!));
    b_f  = GPUTensor<Vector>(extractVector(newWeights['b_f']!));

    K_xi = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xi']!));
    K_hi = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hi']!));
    b_i  = GPUTensor<Vector>(extractVector(newWeights['b_i']!));

    K_xc = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xc']!));
    K_hc = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hc']!));
    b_c  = GPUTensor<Vector>(extractVector(newWeights['b_c']!));

    K_xo = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xo']!));
    K_ho = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_ho']!));
    b_o  = GPUTensor<Vector>(extractVector(newWeights['b_o']!));

    List<double> zeroBias = <double>[];
    for (int i = 0; i < hiddenFilters; i = i + 1) {
      zeroBias.add(0.0);
    }
    zeroBiasInput = GPUTensor<Vector>(zeroBias);

    built = true;
  }
}