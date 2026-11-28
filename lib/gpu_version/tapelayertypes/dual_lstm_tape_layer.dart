import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_aliases.dart';

import '../ffi/allocation_buffer.dart';
import '../ffi/command_buffer.dart';
import 'tape_layer.dart';

/// Advanced Layer to process sequences of data. It is designed to capture short and long term dependencies of data entries.
/// It is given a Matrix in the shape of (Time, Features) and returns a prediction for the next set of features.
/// A Low tier resolution LSTM updates every step to capture short-term dependencies while a long-term high tier resolution layer provides trend prediction.
class DualLSTMTL extends TapeLayer<Matrix, Vector> {
  @override
  String get name => 'DualLSTMTapeLayer';

  int hiddenSize;
  int lowerTierClockCycle;

  /// Forget, input, cell and output weights for the low tier LSTM.
  late GPUTensor<Matrix> lWf, lWi, lWc, lWo;
  late GPUTensor<Matrix> lbf, lbi, lbc, lbo;

  /// Forget, input, cell and output weights for the high tier LSTM.
  late GPUTensor<Matrix> hWf, hWi, hWc, hWo;
  late GPUTensor<Matrix> hbf, hbi, hbc, hbo;

  /// Requires the amount of hidden features used for both high and low tier LSTM [hiddenSize].
  /// The update rate of the higher tier LSTM in contrast to the low tier is given by [lowerTierClockCycle].
  DualLSTMTL(this.hiddenSize, {this.lowerTierClockCycle = 7});

  /// Returns all trainable weights and biases for the LSTM gates.
  /// Specifically returns for the lower tier: [lWf],[lbf],[lWi],[lbi],[lWc],[lbc],[lWo] and [lbo].
  /// For the high tier:  [hWf],[hbf],[hWi],[hbi],[hWc],[hbc],[hWo] and [hbo].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(<GPUTensor>[lWf, lbf, lWi, lbi, lWc, lbc, lWo, lbo]);
      params.addAll(<GPUTensor>[hWf, hbf, hWi, hbi, hWc, hbc, hWo, hbo]);
    }
    return params;
  }

  /// Allocates VRAM for all kernels and biases. Weights are initialized using scaled random values.
  @override
  void build(GPUTensor<Matrix> input, AllocBuffer allocTape) {
    int inputSize = input.shape[1];

    int lowerCombinedSize = hiddenSize + hiddenSize + inputSize;
    int higherCombinedSize = hiddenSize + hiddenSize;

    Random random = Random();

    List<List<double>> initWeightsTransposed(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      List<List<double>> values = <List<double>>[];
      for (int i = 0; i < fanIn; i = i + 1) {
        List<double> row = <double>[];
        for (int j = 0; j < fanOut; j = j + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        values.add(row);
      }
      return values;
    }

    List<List<double>> initBiasRow(int size) {
      List<double> values = <double>[];
      for (int i = 0; i < size; i = i + 1) {
        values.add(0.0);
      }
      return <List<double>>[values];
    }

    lWf = GPUTensor<Matrix>(
        initWeightsTransposed(lowerCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:lWf');
    lWi = GPUTensor<Matrix>(
        initWeightsTransposed(lowerCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:lWi');
    lWc = GPUTensor<Matrix>(
        initWeightsTransposed(lowerCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:lWc');
    lWo = GPUTensor<Matrix>(
        initWeightsTransposed(lowerCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:lWo');

    lbf = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:lbf');
    lbi = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:lbi');
    lbc = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:lbc');
    lbo = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:lbo');

    hWf = GPUTensor<Matrix>(
        initWeightsTransposed(higherCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:hWf');
    hWi = GPUTensor<Matrix>(
        initWeightsTransposed(higherCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:hWi');
    hWc = GPUTensor<Matrix>(
        initWeightsTransposed(higherCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:hWc');
    hWo = GPUTensor<Matrix>(
        initWeightsTransposed(higherCombinedSize, hiddenSize), allocTape,
        sourceOp: '$name:hWo');

    hbf = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:hbf');
    hbi = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:hbi');
    hbc = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:hbc');
    hbo = GPUTensor<Matrix>(initBiasRow(hiddenSize), allocTape,
        sourceOp: '$name:hbo');

    built = true;
  }

  /// Appends the recurrent sequential tiered LSTM operations to the provided [tape].
  /// Returns the [GPUTensor] representing the final hidden state of the sequence.
  @override
  GPUTensor<Vector> forward(
      GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
    int totalSteps = input.shape[0];
    int inputSize = input.shape[1];

    List<int> stateShape = <int>[1, hiddenSize];
    List<GPUTensor<Matrix>> lhStates = <GPUTensor<Matrix>>[];
    List<GPUTensor<Matrix>> lcStates = <GPUTensor<Matrix>>[];
    List<GPUTensor<Matrix>> hhStates = <GPUTensor<Matrix>>[];
    List<GPUTensor<Matrix>> hcStates = <GPUTensor<Matrix>>[];

    lhStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initLH'));
    lcStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initLC'));
    hhStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initHH'));
    hcStates.add(GPUTensor<Matrix>.empty(stateShape, allocTape,
        sourceOp: '$name:initHC'));

    for (int i = 0; i < totalSteps; i = i + 1) {
      GPUTensor<Vector> xVec = selectRowGPU(input, i, tape, allocTape);
      GPUTensor<Matrix> xT =
      reshapeVectorToMatrixGPU(xVec, 1, inputSize, tape, allocTape);

      GPUTensor<Matrix> prevLH = lhStates[i];
      GPUTensor<Matrix> prevLC = lcStates[i];
      GPUTensor<Matrix> prevHH = hhStates[i];
      GPUTensor<Matrix> prevHC = hcStates[i];

      //  LOWER TIER
      GPUTensor<Matrix> combLow = concatenateMatricesByColumnGPU(
          <GPUTensor<Matrix>>[prevLH, prevHC, xT], tape, allocTape);

      GPUTensor<Matrix> lfLinear = matMulGPU(combLow, lWf, tape, allocTape);
      GPUTensor<Matrix> lfBiased = addMatrixGPU(lfLinear, lbf, tape, allocTape);
      GPUTensor<Matrix> lfT = sigmoidMatrixGPU(lfBiased, tape, allocTape);

      GPUTensor<Matrix> liLinear = matMulGPU(combLow, lWi, tape, allocTape);
      GPUTensor<Matrix> liBiased = addMatrixGPU(liLinear, lbi, tape, allocTape);
      GPUTensor<Matrix> liT = sigmoidMatrixGPU(liBiased, tape, allocTape);

      GPUTensor<Matrix> lcTildeLinear =
      matMulGPU(combLow, lWc, tape, allocTape);
      GPUTensor<Matrix> lcTildeBiased =
      addMatrixGPU(lcTildeLinear, lbc, tape, allocTape);
      GPUTensor<Matrix> lcTilde =
      tanhMatrixGPU(lcTildeBiased, tape, allocTape);

      GPUTensor<Matrix> lcRetained =
      elementWiseMultiplyMatrixGPU(lfT, prevLC, tape, allocTape);
      GPUTensor<Matrix> lcNewInfo =
      elementWiseMultiplyMatrixGPU(liT, lcTilde, tape, allocTape);

      GPUTensor<Matrix> newLC =
      addMatrixGPU(lcRetained, lcNewInfo, tape, allocTape);
      lcStates.add(newLC);

      GPUTensor<Matrix> loLinear = matMulGPU(combLow, lWo, tape, allocTape);
      GPUTensor<Matrix> loBiased = addMatrixGPU(loLinear, lbo, tape, allocTape);
      GPUTensor<Matrix> loT = sigmoidMatrixGPU(loBiased, tape, allocTape);

      GPUTensor<Matrix> lcActivated = tanhMatrixGPU(newLC, tape, allocTape);

      GPUTensor<Matrix> newLH =
      elementWiseMultiplyMatrixGPU(loT, lcActivated, tape, allocTape);
      lhStates.add(newLH);

      //  HIGHER TIER
      if (i > 0 && (i + 1) % lowerTierClockCycle == 0) {
        GPUTensor<Matrix> combHigh = concatenateMatricesByColumnGPU(
            <GPUTensor<Matrix>>[prevHH, newLH], tape, allocTape);

        GPUTensor<Matrix> hfLinear = matMulGPU(combHigh, hWf, tape, allocTape);
        GPUTensor<Matrix> hfBiased =
        addMatrixGPU(hfLinear, hbf, tape, allocTape);
        GPUTensor<Matrix> hfT = sigmoidMatrixGPU(hfBiased, tape, allocTape);

        GPUTensor<Matrix> hiLinear = matMulGPU(combHigh, hWi, tape, allocTape);
        GPUTensor<Matrix> hiBiased =
        addMatrixGPU(hiLinear, hbi, tape, allocTape);
        GPUTensor<Matrix> hiT = sigmoidMatrixGPU(hiBiased, tape, allocTape);

        GPUTensor<Matrix> hcTildeLinear =
        matMulGPU(combHigh, hWc, tape, allocTape);
        GPUTensor<Matrix> hcTildeBiased =
        addMatrixGPU(hcTildeLinear, hbc, tape, allocTape);
        GPUTensor<Matrix> hcTilde =
        tanhMatrixGPU(hcTildeBiased, tape, allocTape);

        GPUTensor<Matrix> hcRetained =
        elementWiseMultiplyMatrixGPU(hfT, prevHC, tape, allocTape);
        GPUTensor<Matrix> hcNewInfo =
        elementWiseMultiplyMatrixGPU(hiT, hcTilde, tape, allocTape);

        GPUTensor<Matrix> newHC =
        addMatrixGPU(hcRetained, hcNewInfo, tape, allocTape);
        hcStates.add(newHC);

        GPUTensor<Matrix> hoLinear = matMulGPU(combHigh, hWo, tape, allocTape);
        GPUTensor<Matrix> hoBiased =
        addMatrixGPU(hoLinear, hbo, tape, allocTape);
        GPUTensor<Matrix> hoT = sigmoidMatrixGPU(hoBiased, tape, allocTape);

        GPUTensor<Matrix> hcActivated = tanhMatrixGPU(newHC, tape, allocTape);

        GPUTensor<Matrix> newHH =
        elementWiseMultiplyMatrixGPU(hoT, hcActivated, tape, allocTape);
        hhStates.add(newHH);
      } else {
        hcStates.add(prevHC);
        hhStates.add(prevHH);
      }
    }

    GPUTensor<Vector> finalVector =
    selectRowGPU(lhStates.last, 0, tape, allocTape);

    return finalVector;
  }

  /// Helper function for storing values in a map in a transposed form.
  List<List<double>> transposeForCpu(List<List<double>> m) {
    int rows = m.length;
    int cols = m[0].length;
    List<List<double>> res = <List<double>>[];
    for (int i = 0; i < cols; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < rows; j = j + 1) {
        row.add(m[j][i]);
      }
      res.add(row);
    }
    return res;
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    lWf.dataToCpu();
    lbf.dataToCpu();
    lWi.dataToCpu();
    lbi.dataToCpu();
    lWc.dataToCpu();
    lbc.dataToCpu();
    lWo.dataToCpu();
    lbo.dataToCpu();
    hWf.dataToCpu();
    hbf.dataToCpu();
    hWi.dataToCpu();
    hbi.dataToCpu();
    hWc.dataToCpu();
    hbc.dataToCpu();
    hWo.dataToCpu();
    hbo.dataToCpu();

    wMap['lW_f'] = transposeForCpu(lWf.value);
    wMap['lW_i'] = transposeForCpu(lWi.value);
    wMap['lW_c'] = transposeForCpu(lWc.value);
    wMap['lW_o'] = transposeForCpu(lWo.value);
    wMap['hW_f'] = transposeForCpu(hWf.value);
    wMap['hW_i'] = transposeForCpu(hWi.value);
    wMap['hW_c'] = transposeForCpu(hWc.value);
    wMap['hW_o'] = transposeForCpu(hWo.value);

    wMap['lb_f'] = lbf.value[0];
    wMap['lb_i'] = lbi.value[0];
    wMap['lb_c'] = lbc.value[0];
    wMap['lb_o'] = lbo.value[0];
    wMap['hb_f'] = hbf.value[0];
    wMap['hb_i'] = hbi.value[0];
    wMap['hb_c'] = hbc.value[0];
    wMap['hb_o'] = hbo.value[0];

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    List<double> extractAndTransposeFlat(List<dynamic> raw) {
      List<List<double>> m = <List<double>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<double> row = <double>[];
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          row.add(rawRow[j] as double);
        }
        m.add(row);
      }
      List<List<double>> transposed = transposeForCpu(m);
      List<double> flat = <double>[];
      for (int i = 0; i < transposed.length; i = i + 1) {
        for (int j = 0; j < transposed[i].length; j = j + 1) {
          flat.add(transposed[i][j]);
        }
      }
      return flat;
    }

    List<double> extractVector(List<dynamic> raw) {
      List<double> res = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        res.add(raw[i] as double);
      }
      return res;
    }

    lWf.pushData(extractAndTransposeFlat(newWeights['lW_f']!));
    lWi.pushData(extractAndTransposeFlat(newWeights['lW_i']!));
    lWc.pushData(extractAndTransposeFlat(newWeights['lW_c']!));
    lWo.pushData(extractAndTransposeFlat(newWeights['lW_o']!));

    lbf.pushData(extractVector(newWeights['lb_f']!));
    lbi.pushData(extractVector(newWeights['lb_i']!));
    lbc.pushData(extractVector(newWeights['lb_c']!));
    lbo.pushData(extractVector(newWeights['lb_o']!));

    hWf.pushData(extractAndTransposeFlat(newWeights['hW_f']!));
    hWi.pushData(extractAndTransposeFlat(newWeights['hW_i']!));
    hWc.pushData(extractAndTransposeFlat(newWeights['hW_c']!));
    hWo.pushData(extractAndTransposeFlat(newWeights['hW_o']!));

    hbf.pushData(extractVector(newWeights['hb_f']!));
    hbi.pushData(extractVector(newWeights['hb_i']!));
    hbc.pushData(extractVector(newWeights['hb_c']!));
    hbo.pushData(extractVector(newWeights['hb_o']!));
  }
}