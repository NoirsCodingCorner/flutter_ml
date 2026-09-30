import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/op_codes.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Advanced Layer to process sequences of 2D spatial data. In this case it is represented as
/// Tensor3Ds with format 'Time, Height, Width'. It uses convolutional gates instead of standard matrix multiplications.
/// Therefore this architecture is able to capture both time and spatial information.
class ConvLSTMTL extends TapeLayer<Tensor3D, Tensor3D> {
  @override
  String get name => 'ConvLSTMTapeLayer';

  int hiddenFilters;
  int kernelSize;
  double gradClipValue;

  // Forget gate
  /// Learnable input convolution kernel for the forget gate.
  late GPUTensor<Tensor3D> kXf;

  /// Learnable hidden state convolution kernel for the forget gate.
  late GPUTensor<Tensor3D> kHf;

  // Input gate
  /// Learnable input convolution kernel for the input gate.
  late GPUTensor<Tensor3D> kXi;

  /// Learnable hidden state convolution kernel for the input gate.
  late GPUTensor<Tensor3D> kHi;

  // Cell candidate
  /// Learnable input convolution kernel for the cell candidate.
  late GPUTensor<Tensor3D> kXc;

  /// Learnable hidden state convolution kernel for the cell candidate.
  late GPUTensor<Tensor3D> kHc;

  // Output gate
  /// Learnable input convolution kernel for the output gate.
  late GPUTensor<Tensor3D> kXo;

  /// Learnable hidden state convolution kernel for the output gate.
  late GPUTensor<Tensor3D> kHo;

  /// Learnable bias vector for the forget gate.
  late GPUTensor<Vector> bf;

  /// Learnable bias vector for the input gate.
  late GPUTensor<Vector> bi;

  /// Learnable bias vector for the cell candidate.
  late GPUTensor<Vector> bc;

  /// Learnable bias vector for the output gate.
  late GPUTensor<Vector> bo;

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
  /// Specifically returns [kXf],[kHf],[bf],[kXi],[kHi],[bi],[kXc],[kHc],[bc],[kXo],[kHo] and [bo].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(kXf);
      params.add(kHf);
      params.add(bf);
      params.add(kXi);
      params.add(kHi);
      params.add(bi);
      params.add(kXc);
      params.add(kHc);
      params.add(bc);
      params.add(kXo);
      params.add(kHo);
      params.add(bo);
    }
    return params;
  }

  /// Allocates VRAM for all kernels and biases. Weights are initialized using scaled random values.
  @override
  void build(GPUTensor<dynamic> input) {
    int inChannels = 1;
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

    kXf = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    kHf = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    kXi = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    kHi = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    kXc = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    kHc = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));
    kXo = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters));
    kHo = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters));

    bf = GPUTensor<Vector>(initBias(hiddenFilters));
    bi = GPUTensor<Vector>(initBias(hiddenFilters));
    bc = GPUTensor<Vector>(initBias(hiddenFilters));
    bo = GPUTensor<Vector>(initBias(hiddenFilters));

    zeroBiasInput = GPUTensor<Vector>(initBias(hiddenFilters));

    built = true;
  }

  /// Appends the recurrent sequential convolution operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  /// To support static tape unrolling, this layer now persistently caches its own intermediates
  /// and bypasses the external [intermediates] list. It dynamically reallocates memory only if the sequence length changes.
  @override
  GPUTensor<Tensor3D> forward(GPUTensor<Tensor3D> input, CommandBuffer tape,
      List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];

    bool useCache = (cacheSeqLength == seqLength);

    // If sequence length changed or this is the first run, rebuild the cache.
    if (!useCache) {
      // Free old VRAM allocations to prevent memory leaks if sequence length changed dynamically
      for (int i = 0; i < hStates.length; i = i + 1) {
        hStates[i].free();
      }
      for (int i = 0; i < cStates.length; i = i + 1) {
        cStates[i].free();
      }
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }

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
      GPUTensor<Matrix> xT = selectMatrixFrom3DGPU(input, t, tape,
          outTensor: getCached<GPUTensor<Matrix>>());
      saveCached(xT);

      // Explicitly reference the previous timestep's states
      GPUTensor<Tensor3D> prevH = hStates[t];
      GPUTensor<Tensor3D> prevC = cStates[t];

      // Forget Gate
      GPUTensor<Tensor3D> fX = conv2dMultiChannelGPU(
          xT, kXf, bf, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fX);
      GPUTensor<Tensor3D> fH = conv2dMultiChannelGPU(
          prevH, kHf, zeroBiasInput, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fH);
      GPUTensor<Tensor3D> fSum =
          add3DGPU(fX, fH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fSum);
      GPUTensor<Tensor3D> fT =
          sigmoid3DGPU(fSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(fT);

      // Input Gate
      GPUTensor<Tensor3D> iX = conv2dMultiChannelGPU(
          xT, kXi, bi, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iX);
      GPUTensor<Tensor3D> iH = conv2dMultiChannelGPU(
          prevH, kHi, zeroBiasInput, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iH);
      GPUTensor<Tensor3D> iSum =
          add3DGPU(iX, iH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iSum);
      GPUTensor<Tensor3D> iT =
          sigmoid3DGPU(iSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(iT);

      // Cell Candidate
      GPUTensor<Tensor3D> cX = conv2dMultiChannelGPU(
          xT, kXc, bc, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cX);
      GPUTensor<Tensor3D> cH = conv2dMultiChannelGPU(
          prevH, kHc, zeroBiasInput, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cH);
      GPUTensor<Tensor3D> cSum =
          add3DGPU(cX, cH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cSum);
      GPUTensor<Tensor3D> cTildeT =
          tanh3DGPU(cSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cTildeT);

      // Cell State Update
      GPUTensor<Tensor3D> cRetained = elementWiseMultiply3DGPU(fT, prevC, tape,
          outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cRetained);
      GPUTensor<Tensor3D> cNewInfo = elementWiseMultiply3DGPU(iT, cTildeT, tape,
          outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cNewInfo);

      GPUTensor<Tensor3D>? outNewC = useCache ? cStates[t + 1] : null;
      GPUTensor<Tensor3D> newC =
          add3DGPU(cRetained, cNewInfo, tape, outTensor: outNewC);
      if (!useCache) cStates.add(newC);

      // Output Gate
      GPUTensor<Tensor3D> oX = conv2dMultiChannelGPU(
          xT, kXo, bo, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oX);
      GPUTensor<Tensor3D> oH = conv2dMultiChannelGPU(
          prevH, kHo, zeroBiasInput, kernelSize, kernelSize, tape,
          padding: 'same', outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oH);
      GPUTensor<Tensor3D> oSum =
          add3DGPU(oX, oH, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oSum);
      GPUTensor<Tensor3D> oT =
          sigmoid3DGPU(oSum, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(oT);

      // Hidden State Update
      GPUTensor<Tensor3D> cActivated =
          tanh3DGPU(newC, tape, outTensor: getCached<GPUTensor<Tensor3D>>());
      saveCached(cActivated);

      GPUTensor<Tensor3D>? outNewH = useCache ? hStates[t + 1] : null;
      GPUTensor<Tensor3D> newH =
          elementWiseMultiply3DGPU(oT, cActivated, tape, outTensor: outNewH);
      if (!useCache) hStates.add(newH);
    }

    GPUTensor<Tensor3D> finalH = hStates.last;

    GPUNode? originalCreator = finalH.creator;
    if (originalCreator != null) {
      void Function(CommandBuffer) originalBackward =
          originalCreator.backwardFn;
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
      kXf.free();
      kHf.free();
      kXi.free();
      kHi.free();
      kXc.free();
      kHc.free();
      kXo.free();
      kHo.free();
      bf.free();
      bi.free();
      bc.free();
      bo.free();
      zeroBiasInput.free();

      // Free persistently cached states and intermediates
      for (int i = 0; i < hStates.length; i = i + 1) {
        hStates[i].free();
      }
      for (int i = 0; i < cStates.length; i = i + 1) {
        cStates[i].free();
      }
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
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
    if (built == false) {
      return wMap;
    }

    kXf.toCpu();
    kHf.toCpu();
    bf.toCpu();
    kXi.toCpu();
    kHi.toCpu();
    bi.toCpu();
    kXc.toCpu();
    kHc.toCpu();
    bc.toCpu();
    kXo.toCpu();
    kHo.toCpu();
    bo.toCpu();

    wMap['K_xf'] = kXf.value;
    wMap['K_hf'] = kHf.value;
    wMap['b_f'] = bf.value;
    wMap['K_xi'] = kXi.value;
    wMap['K_hi'] = kHi.value;
    wMap['b_i'] = bi.value;
    wMap['K_xc'] = kXc.value;
    wMap['K_hc'] = kHc.value;
    wMap['b_c'] = bc.value;
    wMap['K_xo'] = kXo.value;
    wMap['K_ho'] = kHo.value;
    wMap['b_o'] = bo.value;

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

    kXf = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xf']!));
    kHf = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hf']!));
    bf = GPUTensor<Vector>(extractVector(newWeights['b_f']!));

    kXi = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xi']!));
    kHi = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hi']!));
    bi = GPUTensor<Vector>(extractVector(newWeights['b_i']!));

    kXc = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xc']!));
    kHc = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_hc']!));
    bc = GPUTensor<Vector>(extractVector(newWeights['b_c']!));

    kXo = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_xo']!));
    kHo = GPUTensor<Tensor3D>(extractTensor3D(newWeights['K_ho']!));
    bo = GPUTensor<Vector>(extractVector(newWeights['b_o']!));

    List<double> zeroBias = <double>[];
    for (int i = 0; i < hiddenFilters; i = i + 1) {
      zeroBias.add(0.0);
    }
    zeroBiasInput = GPUTensor<Vector>(zeroBias);

    built = true;
  }
}
