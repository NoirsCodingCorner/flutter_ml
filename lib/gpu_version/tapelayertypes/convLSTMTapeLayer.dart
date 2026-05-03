import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/OpCodes.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class ConvLSTMTL extends TapeLayer {
  int hiddenFilters;
  int kernelSize;
  double gradClipValue;

  late GPUTensor<Tensor3D> K_xf;
  late GPUTensor<Tensor3D> K_hf;
  late GPUTensor<Tensor3D> K_xi;
  late GPUTensor<Tensor3D> K_hi;
  late GPUTensor<Tensor3D> K_xc;
  late GPUTensor<Tensor3D> K_hc;
  late GPUTensor<Tensor3D> K_xo;
  late GPUTensor<Tensor3D> K_ho;

  late GPUTensor<Vector> b_f;
  late GPUTensor<Vector> b_i;
  late GPUTensor<Vector> b_c;
  late GPUTensor<Vector> b_o;
  late GPUTensor<Vector> zeroBiasInput;

  ConvLSTMTL(this.hiddenFilters, this.kernelSize, {this.gradClipValue = 1.0});

  @override
  String get name {
    return 'ConvLSTMTapeLayer';
  }

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

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Tensor3D> sequence = input as GPUTensor<Tensor3D>;
    int seqLength = sequence.shape[0];
    int height = sequence.shape[1];
    int width = sequence.shape[2];

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

    GPUTensor<Tensor3D> h = GPUTensor<Tensor3D>(zero3D);
    GPUTensor<Tensor3D> c = GPUTensor<Tensor3D>(zero3D);

    intermediates.add(h);
    intermediates.add(c);

    for (int t = 0; t < seqLength; t = t + 1) {
      GPUTensor<Matrix> xT = selectMatrixFrom3DGPU(sequence, t, tape);
      intermediates.add(xT);

      // Forget Gate
      GPUTensor<Tensor3D> fX = conv2dMultiChannelGPU(xT, K_xf, b_f, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> fH = conv2dMultiChannelGPU(h, K_hf, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> fSum = add3DGPU(fX, fH, tape);
      GPUTensor<Tensor3D> fT = sigmoid3DGPU(fSum, tape);

      intermediates.add(fX);
      intermediates.add(fH);
      intermediates.add(fSum);
      intermediates.add(fT);

      // Input Gate
      GPUTensor<Tensor3D> iX = conv2dMultiChannelGPU(xT, K_xi, b_i, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> iH = conv2dMultiChannelGPU(h, K_hi, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> iSum = add3DGPU(iX, iH, tape);
      GPUTensor<Tensor3D> iT = sigmoid3DGPU(iSum, tape);

      intermediates.add(iX);
      intermediates.add(iH);
      intermediates.add(iSum);
      intermediates.add(iT);

      // Cell Candidate
      GPUTensor<Tensor3D> cX = conv2dMultiChannelGPU(xT, K_xc, b_c, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> cH = conv2dMultiChannelGPU(h, K_hc, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> cSum = add3DGPU(cX, cH, tape);
      GPUTensor<Tensor3D> cTildeT = tanh3DGPU(cSum, tape);

      intermediates.add(cX);
      intermediates.add(cH);
      intermediates.add(cSum);
      intermediates.add(cTildeT);

      // Cell State Update
      GPUTensor<Tensor3D> cRetained = elementWiseMultiply3DGPU(fT, c, tape);
      GPUTensor<Tensor3D> cNewInfo = elementWiseMultiply3DGPU(iT, cTildeT, tape);
      c = add3DGPU(cRetained, cNewInfo, tape);

      intermediates.add(cRetained);
      intermediates.add(cNewInfo);
      if (t < seqLength - 1) {
        intermediates.add(c);
      }

      // Output Gate
      GPUTensor<Tensor3D> oX = conv2dMultiChannelGPU(xT, K_xo, b_o, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> oH = conv2dMultiChannelGPU(h, K_ho, zeroBiasInput, kernelSize, kernelSize, tape, padding: 'same');
      GPUTensor<Tensor3D> oSum = add3DGPU(oX, oH, tape);
      GPUTensor<Tensor3D> oT = sigmoid3DGPU(oSum, tape);

      intermediates.add(oX);
      intermediates.add(oH);
      intermediates.add(oSum);
      intermediates.add(oT);

      // Hidden State Update
      GPUTensor<Tensor3D> cActivated = tanh3DGPU(c, tape);
      h = elementWiseMultiply3DGPU(oT, cActivated, tape);

      intermediates.add(cActivated);
      if (t < seqLength - 1) {
        intermediates.add(h);
      }
    }

    GPUNode? originalCreator = h.creator;
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

    return h;
  }

  @override
  void free() {
    if (built) {
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