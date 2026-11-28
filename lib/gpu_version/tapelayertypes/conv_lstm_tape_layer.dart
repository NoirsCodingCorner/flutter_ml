import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
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
  void build(GPUTensor<Tensor3D> input, AllocBuffer allocTape) {
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

    kXf = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters), allocTape,
        sourceOp: '$name:kXf');
    kHf = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters), allocTape,
        sourceOp: '$name:kHf');
    kXi = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters), allocTape,
        sourceOp: '$name:kXi');
    kHi = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters), allocTape,
        sourceOp: '$name:kHi');
    kXc = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters), allocTape,
        sourceOp: '$name:kXc');
    kHc = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters), allocTape,
        sourceOp: '$name:kHc');
    kXo = GPUTensor<Tensor3D>(initWeight(inChannels, hiddenFilters), allocTape,
        sourceOp: '$name:kXo');
    kHo = GPUTensor<Tensor3D>(initWeight(hiddenFilters, hiddenFilters), allocTape,
        sourceOp: '$name:kHo');

    bf = GPUTensor<Vector>(initBias(hiddenFilters), allocTape,
        sourceOp: '$name:bf');
    bi = GPUTensor<Vector>(initBias(hiddenFilters), allocTape,
        sourceOp: '$name:bi');
    bc = GPUTensor<Vector>(initBias(hiddenFilters), allocTape,
        sourceOp: '$name:bc');
    bo = GPUTensor<Vector>(initBias(hiddenFilters), allocTape,
        sourceOp: '$name:bo');

    zeroBiasInput = GPUTensor<Vector>(initBias(hiddenFilters), allocTape,
        requiresGrad: false, sourceOp: '$name:zeroBiasInput');

    built = true;
  }

  /// Appends the recurrent sequential convolution operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  @override
  GPUTensor<Tensor3D> forward(
      GPUTensor<Tensor3D> input, CommandBuffer tape, AllocBuffer allocTape) {
    int seqLength = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];

    List<int> stateShape = <int>[hiddenFilters, height, width];
    List<GPUTensor<Tensor3D>> hStates = <GPUTensor<Tensor3D>>[];
    List<GPUTensor<Tensor3D>> cStates = <GPUTensor<Tensor3D>>[];

    hStates.add(GPUTensor<Tensor3D>.empty(stateShape, allocTape,
        sourceOp: '$name:initH'));
    cStates.add(GPUTensor<Tensor3D>.empty(stateShape, allocTape,
        sourceOp: '$name:initC'));

    for (int t = 0; t < seqLength; t = t + 1) {
      GPUTensor<Matrix> xT =
      selectMatrixFrom3DGPU(input, t, tape, allocTape);

      GPUTensor<Tensor3D> prevH = hStates[t];
      GPUTensor<Tensor3D> prevC = cStates[t];

      // Forget Gate
      GPUTensor<Tensor3D> fX = conv2dMultiChannelGPU(
          xT, kXf, bf, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> fH = conv2dMultiChannelGPU(
          prevH, kHf, zeroBiasInput, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> fSum = add3DGPU(fX, fH, tape, allocTape);
      GPUTensor<Tensor3D> fT = sigmoid3DGPU(fSum, tape, allocTape);

      // Input Gate
      GPUTensor<Tensor3D> iX = conv2dMultiChannelGPU(
          xT, kXi, bi, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> iH = conv2dMultiChannelGPU(
          prevH, kHi, zeroBiasInput, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> iSum = add3DGPU(iX, iH, tape, allocTape);
      GPUTensor<Tensor3D> iT = sigmoid3DGPU(iSum, tape, allocTape);

      // Cell Candidate
      GPUTensor<Tensor3D> cX = conv2dMultiChannelGPU(
          xT, kXc, bc, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> cH = conv2dMultiChannelGPU(
          prevH, kHc, zeroBiasInput, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> cSum = add3DGPU(cX, cH, tape, allocTape);
      GPUTensor<Tensor3D> cTildeT = tanh3DGPU(cSum, tape, allocTape);

      // Cell State Update
      GPUTensor<Tensor3D> cRetained =
      elementWiseMultiply3DGPU(fT, prevC, tape, allocTape);
      GPUTensor<Tensor3D> cNewInfo =
      elementWiseMultiply3DGPU(iT, cTildeT, tape, allocTape);

      GPUTensor<Tensor3D> newC = add3DGPU(cRetained, cNewInfo, tape, allocTape);
      cStates.add(newC);

      // Output Gate
      GPUTensor<Tensor3D> oX = conv2dMultiChannelGPU(
          xT, kXo, bo, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> oH = conv2dMultiChannelGPU(
          prevH, kHo, zeroBiasInput, kernelSize, kernelSize, tape, allocTape,
          padding: 'same');
      GPUTensor<Tensor3D> oSum = add3DGPU(oX, oH, tape, allocTape);
      GPUTensor<Tensor3D> oT = sigmoid3DGPU(oSum, tape, allocTape);

      // Hidden State Update
      GPUTensor<Tensor3D> cActivated = tanh3DGPU(newC, tape, allocTape);

      GPUTensor<Tensor3D> newH =
      elementWiseMultiply3DGPU(oT, cActivated, tape, allocTape);
      hStates.add(newH);
    }

    GPUTensor<Tensor3D> finalH = hStates.last;

    GPUNode? originalCreator = finalH.creator;
    if (originalCreator != null) {
      void Function(CommandBuffer) originalBackward =
          originalCreator.backwardFn;
      originalCreator.backwardFn = (CommandBuffer bTape) {
        originalBackward(bTape);
        List<GPUTensor> params = parameters;
        for (int i = 0; i < params.length; i = i + 1) {
          bTape.putInt(OP_CLIP_GRAD_VALUE);
          bTape.putString('${params[i].id}_grad');
          bTape.putPatchableFloat('clip_value', gradClipValue);
        }
      };
    }

    return finalH;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    kXf.dataToCpu();
    kHf.dataToCpu();
    bf.dataToCpu();
    kXi.dataToCpu();
    kHi.dataToCpu();
    bi.dataToCpu();
    kXc.dataToCpu();
    kHc.dataToCpu();
    bc.dataToCpu();
    kXo.dataToCpu();
    kHo.dataToCpu();
    bo.dataToCpu();

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
    List<double> extractVector(List<dynamic> raw) {
      List<double> result = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        result.add(raw[i] as double);
      }
      return result;
    }

    List<double> extractFlatTensor3D(List<dynamic> raw) {
      List<double> result = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<dynamic> rawChannel = raw[i] as List<dynamic>;
        for (int j = 0; j < rawChannel.length; j = j + 1) {
          List<dynamic> rawRow = rawChannel[j] as List<dynamic>;
          for (int k = 0; k < rawRow.length; k = k + 1) {
            result.add(rawRow[k] as double);
          }
        }
      }
      return result;
    }

    kXf.pushData(extractFlatTensor3D(newWeights['K_xf']!));
    kHf.pushData(extractFlatTensor3D(newWeights['K_hf']!));
    bf.pushData(extractVector(newWeights['b_f']!));

    kXi.pushData(extractFlatTensor3D(newWeights['K_xi']!));
    kHi.pushData(extractFlatTensor3D(newWeights['K_hi']!));
    bi.pushData(extractVector(newWeights['b_i']!));

    kXc.pushData(extractFlatTensor3D(newWeights['K_xc']!));
    kHc.pushData(extractFlatTensor3D(newWeights['K_hc']!));
    bc.pushData(extractVector(newWeights['b_c']!));

    kXo.pushData(extractFlatTensor3D(newWeights['K_xo']!));
    kHo.pushData(extractFlatTensor3D(newWeights['K_ho']!));
    bo.pushData(extractVector(newWeights['b_o']!));
  }
}
