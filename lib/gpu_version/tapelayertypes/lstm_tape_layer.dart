import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/op_codes.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Standard Long Short-Term Memory (LSTM) Layer.
/// Processes a sequence of data to capture long-term dependencies.
class LSTMTL extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'LSTMTapeLayer';

  int hiddenSize;
  double gradClipValue;

  late GPUTensor<Matrix> wXf;
  late GPUTensor<Matrix> wHf;
  late GPUTensor<Matrix> bf;
  late GPUTensor<Matrix> wXi;
  late GPUTensor<Matrix> wHi;
  late GPUTensor<Matrix> bi;
  late GPUTensor<Matrix> wXc;
  late GPUTensor<Matrix> wHc;
  late GPUTensor<Matrix> bc;
  late GPUTensor<Matrix> wXo;
  late GPUTensor<Matrix> wHo;
  late GPUTensor<Matrix> bo;

  /// Requires the number of hidden features in both the cell state and the hidden state [hiddenSize].
  /// Optionally a [gradClipValue] can be assigned to prevent gradient explosion.
  LSTMTL(this.hiddenSize, {this.gradClipValue = 1.0});

  /// Returns all trainable parameters of the layer.
  /// Specifically returns [wXf],[wHf],[bf],[wXi],[wHi],[bi],[wXc],[wHc],[bc],[wXo],[wHo] and [bo].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(<GPUTensor>[
        wXf,
        wHf,
        bf,
        wXi,
        wHi,
        bi,
        wXc,
        wHc,
        bc,
        wXo,
        wHo,
        bo,
      ]);
    }
    return params;
  }

  /// Allocates VRAM for all gates and initializes them with random values.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
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

    wXf = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize), allocTape,
        sourceOp: '$name:wXf');
    wHf = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize), allocTape,
        sourceOp: '$name:wHf');
    wXi = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize), allocTape,
        sourceOp: '$name:wXi');
    wHi = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize), allocTape,
        sourceOp: '$name:wHi');
    wXc = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize), allocTape,
        sourceOp: '$name:wXc');
    wHc = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize), allocTape,
        sourceOp: '$name:wHc');
    wXo = GPUTensor<Matrix>(initWeights(inputSize, hiddenSize), allocTape,
        sourceOp: '$name:wXo');
    wHo = GPUTensor<Matrix>(initWeights(hiddenSize, hiddenSize), allocTape,
        sourceOp: '$name:wHo');

    bf = GPUTensor<Matrix>(initBias(), allocTape, sourceOp: '$name:bf');
    bi = GPUTensor<Matrix>(initBias(), allocTape, sourceOp: '$name:bi');
    bc = GPUTensor<Matrix>(initBias(), allocTape, sourceOp: '$name:bc');
    bo = GPUTensor<Matrix>(initBias(), allocTape, sourceOp: '$name:bo');

    built = true;
  }

  // Helper to strictly slice a sequence row into a 1xN Matrix.
  GPUTensor<Matrix> sliceRowToMatrix(
    GPUTensor<Matrix> sequence,
    int rowIdx,
    CommandBuffer tape,
    AllocBuffer allocTape,
  ) {
    GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
        <int>[1, sequence.shape[1]], allocTape,
        sourceOp: '$name:sliceRowToMatrix');
    tape.putInt(OP_SLICE_ROW);
    tape.putString(sequence.id);
    tape.putString(out.id);
    tape.putInt(rowIdx);

    out.creator = GPUNode(<GPUTensor>[sequence], (CommandBuffer bTape) {
      bTape.putInt(OP_SLICE_ROW_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${sequence.id}_grad');
      bTape.putInt(rowIdx);
    }, opName: 'slice_row_matrix_gpu');

    return out;
  }

  /// Appends the recurrent sequential operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  @override
  GPUTensor<Matrix> forward(
    GPUTensor<Matrix> input,
    CommandBuffer tape,
    AllocBuffer allocTape,
  ) {
    int seqLength = input.shape[0];
    List<int> stateShape = <int>[1, hiddenSize];

    List<GPUTensor<Matrix>> hStates = <GPUTensor<Matrix>>[];
    List<GPUTensor<Matrix>> cStates = <GPUTensor<Matrix>>[];

    hStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initH'));
    cStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initC'));

    for (int t = 0; t < seqLength; t = t + 1) {
      GPUTensor<Matrix> xT = sliceRowToMatrix(input, t, tape, allocTape);

      GPUTensor<Matrix> prevH = hStates[t];
      GPUTensor<Matrix> prevC = cStates[t];

      // Forget Gate
      GPUTensor<Matrix> fTX = matMulGPU(xT, wXf, tape, allocTape);
      GPUTensor<Matrix> fTH = matMulGPU(prevH, wHf, tape, allocTape);
      GPUTensor<Matrix> fTSum = addMatrixGPU(fTX, fTH, tape, allocTape);
      GPUTensor<Matrix> fTBiased = addMatrixGPU(fTSum, bf, tape, allocTape);
      GPUTensor<Matrix> fT = sigmoidMatrixGPU(fTBiased, tape, allocTape);

      // Input Gate
      GPUTensor<Matrix> iTX = matMulGPU(xT, wXi, tape, allocTape);
      GPUTensor<Matrix> iTH = matMulGPU(prevH, wHi, tape, allocTape);
      GPUTensor<Matrix> iTSum = addMatrixGPU(iTX, iTH, tape, allocTape);
      GPUTensor<Matrix> iTBiased = addMatrixGPU(iTSum, bi, tape, allocTape);
      GPUTensor<Matrix> iT = sigmoidMatrixGPU(iTBiased, tape, allocTape);

      // Cell Candidate
      GPUTensor<Matrix> cTildeTX = matMulGPU(xT, wXc, tape, allocTape);
      GPUTensor<Matrix> cTildeTH = matMulGPU(prevH, wHc, tape, allocTape);
      GPUTensor<Matrix> cTildeTSum =
          addMatrixGPU(cTildeTX, cTildeTH, tape, allocTape);
      GPUTensor<Matrix> cTildeTBiased =
          addMatrixGPU(cTildeTSum, bc, tape, allocTape);
      GPUTensor<Matrix> cTildeT = tanhMatrixGPU(cTildeTBiased, tape, allocTape);

      // Cell State Update
      GPUTensor<Matrix> cRetained =
          elementWiseMultiplyMatrixGPU(fT, prevC, tape, allocTape);
      GPUTensor<Matrix> cNewInfo =
          elementWiseMultiplyMatrixGPU(iT, cTildeT, tape, allocTape);

      GPUTensor<Matrix> newC =
          addMatrixGPU(cRetained, cNewInfo, tape, allocTape);
      cStates.add(newC);

      // Output Gate
      GPUTensor<Matrix> oTX = matMulGPU(xT, wXo, tape, allocTape);
      GPUTensor<Matrix> oTH = matMulGPU(prevH, wHo, tape, allocTape);
      GPUTensor<Matrix> oTSum = addMatrixGPU(oTX, oTH, tape, allocTape);
      GPUTensor<Matrix> oTBiased = addMatrixGPU(oTSum, bo, tape, allocTape);
      GPUTensor<Matrix> oT = sigmoidMatrixGPU(oTBiased, tape, allocTape);

      // Hidden State Update
      GPUTensor<Matrix> cActivated = tanhMatrixGPU(newC, tape, allocTape);

      GPUTensor<Matrix> newH =
          elementWiseMultiplyMatrixGPU(oT, cActivated, tape, allocTape);
      hStates.add(newH);
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
          bTape.putPatchableFloat('clip_value', gradClipValue);
        }
      };
    }

    return finalH;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    wXf.dataToCpu();
    wHf.dataToCpu();
    bf.dataToCpu();
    wXi.dataToCpu();
    wHi.dataToCpu();
    bi.dataToCpu();
    wXc.dataToCpu();
    wHc.dataToCpu();
    bc.dataToCpu();
    wXo.dataToCpu();
    wHo.dataToCpu();
    bo.dataToCpu();

    wMap['W_xf'] = wXf.value;
    wMap['W_hf'] = wHf.value;
    wMap['b_f'] = bf.value;
    wMap['W_xi'] = wXi.value;
    wMap['W_hi'] = wHi.value;
    wMap['b_i'] = bi.value;
    wMap['W_xc'] = wXc.value;
    wMap['W_hc'] = wHc.value;
    wMap['b_c'] = bc.value;
    wMap['W_xo'] = wXo.value;
    wMap['W_ho'] = wHo.value;
    wMap['b_o'] = bo.value;

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

    wXf.pushData(extractFlatMatrix(newWeights['W_xf']!));
    wHf.pushData(extractFlatMatrix(newWeights['W_hf']!));
    bf.pushData(extractFlatMatrix(newWeights['b_f']!));

    wXi.pushData(extractFlatMatrix(newWeights['W_xi']!));
    wHi.pushData(extractFlatMatrix(newWeights['W_hi']!));
    bi.pushData(extractFlatMatrix(newWeights['b_i']!));

    wXc.pushData(extractFlatMatrix(newWeights['W_xc']!));
    wHc.pushData(extractFlatMatrix(newWeights['W_hc']!));
    bc.pushData(extractFlatMatrix(newWeights['b_c']!));

    wXo.pushData(extractFlatMatrix(newWeights['W_xo']!));
    wHo.pushData(extractFlatMatrix(newWeights['W_ho']!));
    bo.pushData(extractFlatMatrix(newWeights['b_o']!));
  }
}
