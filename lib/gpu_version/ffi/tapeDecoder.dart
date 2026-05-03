import 'dart:typed_data';
import 'dart:convert';
import '../../logger.dart';
import 'OpCodes.dart';

class TapeDecoder {
  Uint8List tape;
  final ByteData _reader;
  int _offset = 0;

  TapeDecoder(this.tape) : _reader = ByteData.sublistView(tape);

  void decode() {
    Logger.yellow('--- Decoding Execution Tape (${tape.length} bytes) ---', prefix: '📜');
    _offset = 0;

    while (_offset < tape.length) {
      int opCode = _readInt();
      String opName = _getOpName(opCode);

      switch (opCode) {
      // --- Data Loading & Storage ---
        case OP_LOAD_SAMPLE:
          String nameIn = _readString();
          String nameOut = _readString();
          int sampleIdx = _readInt();
          Logger.cyan('$opName: $nameOut = load_sample($nameIn, index: $sampleIdx)');
          break;
        case OP_STORE_SAMPLE:
          String nameIn = _readString();
          String nameDest = _readString();
          int sampleIdx = _readInt();
          Logger.cyan('$opName: store_sample($nameIn -> $nameDest, index: $sampleIdx)');
          break;
        case OP_COPY:
          String nameIn = _readString();
          String nameOut = _readString();
          Logger.cyan('$opName: $nameOut = copy($nameIn)');
          break;
        case OP_FILL:
          String nameOut = _readString();
          double value = _readFloat();
          Logger.log('$opName: fill($nameOut, $value)');
          break;
        case OP_ZERO_GRAD:
          String nameZero = _readString();
          Logger.yellow('$opName: zero_grad($nameZero)');
          break;

      // --- Basic Arithmetic ---
        case OP_ADD:
        case OP_SUBTRACT:
        case OP_MULTIPLY:
        case OP_DIVIDE:
        case OP_BROADCAST_ADD:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          Logger.cyan('$opName: $nameC = $nameA op $nameB');
          break;
        case OP_ADD_INTO:
        case OP_SUBTRACT_INTO:
          String nameSrc = _readString();
          String nameDest = _readString();
          Logger.cyan('$opName: $nameDest += $nameSrc');
          break;
        case OP_MULTIPLY_BACKWARD:
          String gradOut = _readString();
          String otherIn = _readString();
          String gradIn = _readString();
          Logger.blue('$opName: $gradIn += $gradOut * $otherIn');
          break;
        case OP_DIVIDE_BACKWARD:
          String nameA = _readString();
          String nameB = _readString();
          String gradOut = _readString();
          String gradA = _readString();
          String gradB = _readString();
          Logger.blue('$opName: gradA=$gradA, gradB=$gradB from div($nameA, $nameB, $gradOut)');
          break;
        case OP_ADD_SCALAR:
          String nameInAddS = _readString();
          String nameOutAddS = _readString();
          double scalarAdd = _readFloat();
          Logger.cyan('$opName: $nameOutAddS = $nameInAddS + $scalarAdd');
          break;

      // --- New Element-wise Math ---
        case OP_LOG_ELEMENTWISE:
        case OP_ABS_ELEMENTWISE:
        case OP_SQRT_ELEMENTWISE:
        case OP_EXP_ELEMENTWISE:
          String nameInMath = _readString();
          String nameOutMath = _readString();
          Logger.cyan('$opName: $nameOutMath = op($nameInMath)');
          break;
        case OP_LOG_BACKWARD:
        case OP_ABS_BACKWARD:
          String gradOutMath = _readString();
          String inDataMath = _readString();
          String gradInMath = _readString();
          Logger.blue('$opName: $gradInMath = bw($gradOutMath, inData: $inDataMath)');
          break;
        case OP_EXP_BACKWARD:
        case OP_SQRT_BACKWARD:
          String gradOutSE = _readString();
          String outDataSE = _readString();
          String gradInSE = _readString();
          Logger.blue('$opName: $gradInSE = bw($gradOutSE, outData: $outDataSE)');
          break;
        case OP_POW_ELEMENTWISE:
          String nameInPow = _readString();
          String nameOutPow = _readString();
          double exponent = _readFloat();
          Logger.cyan('$opName: $nameOutPow = pow($nameInPow, exp: $exponent)');
          break;
        case OP_POW_BACKWARD:
          String gradOutPow = _readString();
          String inDataPow = _readString();
          String gradInPow = _readString();
          double expBw = _readFloat();
          Logger.blue('$opName: $gradInPow = pow_bw($gradOutPow, inData: $inDataPow, exp: $expBw)');
          break;
        case OP_CLAMP_ELEMENTWISE:
          String nameInClamp = _readString();
          String nameOutClamp = _readString();
          double minVal = _readFloat();
          double maxVal = _readFloat();
          Logger.cyan('$opName: $nameOutClamp = clamp($nameInClamp, min: $minVal, max: $maxVal)');
          break;
        case OP_CLAMP_BACKWARD:
          String gradOutClamp = _readString();
          String inDataClamp = _readString();
          String gradInClamp = _readString();
          double minValBw = _readFloat();
          double maxValBw = _readFloat();
          Logger.blue('$opName: $gradInClamp = clamp_bw($gradOutClamp, in: $inDataClamp, min: $minValBw, max: $maxValBw)');
          break;

      // --- Matrix Operations ---
        case OP_MATMUL:
          String nameaMm = _readString();
          String namebMm = _readString();
          String namecMm = _readString();
          bool transA = _readBool();
          bool transB = _readBool();
          double alpha = _readFloat();
          double beta = _readFloat();
          bool tCores = _readBool();
          Logger.cyan('$opName: $namecMm = matmul($nameaMm, $namebMm) transA:$transA transB:$transB');
          break;
        case OP_TRANSPOSE:
          String nameInTr = _readString();
          String nameOutTr = _readString();
          Logger.cyan('$opName: $nameOutTr = transpose($nameInTr)');
          break;
        case OP_SCALE_MATRIX:
          String nameInSM = _readString();
          String nameOutSM = _readString();
          double scalarSM = _readFloat();
          Logger.cyan('$opName: $nameOutSM = scale($nameInSM, val: $scalarSM)');
          break;
        case OP_SCALE_MATRIX_BACKWARD:
          String gradOutSMB = _readString();
          String gradInSMB = _readString();
          double scaleSMB = _readFloat();
          Logger.blue('$opName: $gradInSMB += $gradOutSMB * $scaleSMB');
          break;

      // --- Activations ---
        case OP_RELU:
        case OP_SIGMOID:
        case OP_TANH:
        case OP_GELU_FORWARD:
        case OP_SOFTMAX_FORWARD:
          String nameInAct = _readString();
          String nameOutAct = _readString();
          Logger.green('$opName: $nameOutAct = act($nameInAct)');
          break;
        case OP_RELU_BACKWARD:
        case OP_GELU_BACKWARD:
          String nameInBack = _readString();
          String gradOutBack = _readString();
          String gradInBack = _readString();
          Logger.blue('$opName: $gradInBack = act_bw($nameInBack, $gradOutBack)');
          break;
        case OP_SIGMOID_BACKWARD:
        case OP_TANH_BACKWARD:
          String outData = _readString(); // C++ expects outData first here
          String gradOutAct = _readString();
          String gradInAct = _readString();
          Logger.blue('$opName: $gradInAct = act_bw($gradOutAct, outData: $outData)');
          break;
        case OP_SOFTMAX_BACKWARD:
          String gradOutSm = _readString(); // C++ expects gradOut first here
          String outDataSm = _readString();
          String gradInSm = _readString();
          Logger.blue('$opName: $gradInSm = softmax_bw($gradOutSm, outData: $outDataSm)');
          break;

      // --- Loss Functions ---
        case OP_MSE_LOSS_FORWARD:
        case OP_BCE_LOSS_FORWARD:
          String namePred = _readString();
          String nameTarget = _readString();
          String nameOutLoss = _readString();
          Logger.green('$opName: $nameOutLoss = loss($namePred, target: $nameTarget)');
          break;
        case OP_MSE_LOSS_BACKWARD:
        case OP_BCE_LOSS_BACKWARD:
          String gradOutLoss = _readString();
          String namePredLoss = _readString();
          String nameTargetLoss = _readString();
          String gradInLoss = _readString();
          Logger.blue('$opName: $gradInLoss = loss_bw($gradOutLoss, pred: $namePredLoss, target: $nameTargetLoss)');
          break;

      // --- Optimizers ---
        case OP_SGD_UPDATE:
          String nameDataSgd = _readString();
          String nameGradSgd = _readString();
          double lrSgd = _readFloat();
          Logger.yellow('$opName: sgd($nameDataSgd, grad: $nameGradSgd, lr: $lrSgd)');
          break;
        case OP_ADAM_UPDATE:
          String nameDataAdam = _readString();
          String nameGradAdam = _readString();
          String nameM = _readString();
          String nameV = _readString();
          double lrAdam = _readFloat();
          double b1 = _readFloat();
          double b2 = _readFloat();
          double epsAdam = _readFloat();
          int step = _readInt();
          double wd = _readFloat();
          Logger.yellow('$opName: adam($nameDataAdam, step: $step, lr: $lrAdam)');
          break;
        case OP_CLIP_GRAD_VALUE:
          String nameBufferClip = _readString();
          double clipVal = _readFloat();
          Logger.yellow('$opName: clip($nameBufferClip, val: $clipVal)');
          break;

      // --- Reductions ---
        case OP_SUM_REDUCE:
        case OP_SUM_REDUCE_COLUMNS:
        case OP_SUM_REDUCE_ROWS:
          String nameInReduce = _readString();
          String nameOutReduce = _readString();
          Logger.cyan('$opName: $nameOutReduce = reduce($nameInReduce)');
          break;
        case OP_SUM_REDUCE_BACKWARD:
          String gradOutReduce = _readString();
          String gradInReduce = _readString();
          Logger.blue('$opName: $gradInReduce = reduce_bw($gradOutReduce)');
          break;
        case OP_EMBEDDING_FORWARD:
          String nameIndicesEmb = _readString();
          String nameWeightEmb = _readString();
          String nameOutEmb = _readString();
          Logger.cyan('$opName: $nameOutEmb = embedding($nameIndicesEmb, $nameWeightEmb)');
          break;
        case OP_EMBEDDING_BACKWARD:
          String gradOutEmb = _readString();
          String nameIndicesEmbBw = _readString();
          String gradWeightEmb = _readString();
          Logger.blue('$opName: $gradWeightEmb = embedding_bw($gradOutEmb, $nameIndicesEmbBw)');
          break;

      // --- Tensor Manipulation ---
        case OP_SLICE_ROW:
        case OP_SLICE_ROW_BACKWARD:
          String nameInSlice = _readString();
          String nameOutSlice = _readString();
          int row = _readInt();
          Logger.cyan('$opName: $nameOutSlice = slice_row($nameInSlice, row: $row)');
          break;
        case OP_SLICE_COLUMN:
        case OP_SLICE_COLUMN_BACKWARD:
          String nameInCol = _readString();
          String nameOutCol = _readString();
          int start = _readInt();
          int end = _readInt();
          Logger.cyan('$opName: $nameOutCol = slice_col($nameInCol, start: $start, end: $end)');
          break;
        case OP_STACK_ROWS:
          int countStack = _readInt();
          List<String> namesInStack = [];
          for (int i = 0; i < countStack; i = i + 1) {
            namesInStack.add(_readString());
          }
          String nameOutStack = _readString();
          int axisStack = _readInt();
          Logger.cyan('$opName: $nameOutStack = stack(${namesInStack.length} tensors, axis: $axisStack)');
          break;
        case OP_STACK_ROWS_BACKWARD:
          String gradOutStack = _readString();
          int countBw = _readInt();
          List<String> namesGradIn = [];
          for (int i = 0; i < countBw; i = i + 1) {
            namesGradIn.add(_readString());
          }
          int axisBw = _readInt();
          Logger.blue('$opName: bw_stack($gradOutStack -> ${namesGradIn.length} tensors, axis: $axisBw)');
          break;
        case OP_CONCATENATE:
          String nameaCat = _readString();
          String namebCat = _readString();
          String nameOutCat = _readString();
          int axisCat = _readInt();
          Logger.cyan('$opName: $nameOutCat = concat($nameaCat, $namebCat, axis: $axisCat)');
          break;
        case OP_CONCATENATE_BACKWARD:
          String gradOutCat = _readString();
          String gradinaCat = _readString();
          String gradinbCat = _readString();
          int axisCatBw = _readInt();
          int split = _readInt();
          Logger.blue('$opName: bw_concat($gradOutCat -> $gradinaCat, $gradinbCat, split: $split)');
          break;
        case OP_PAD2D:
        case OP_PAD2D_BACKWARD:
          String nameInPad = _readString();
          String nameOutPad = _readString();
          int padT = _readInt();
          int padB = _readInt();
          int padL = _readInt();
          int padR = _readInt();
          Logger.cyan('$opName: $nameOutPad = pad($nameInPad, t:$padT, b:$padB, l:$padL, r:$padR)');
          break;

      // --- Advanced Layers ---
        case OP_MATMUL_BIAS_RELU_FORWARD:
          String nameX = _readString();
          String nameW = _readString();
          String namebMatmul = _readString();
          String nameReluOut = _readString();
          String namePreROut = _readString();
          Logger.cyan('$opName: $nameReluOut = matmul_bias_relu($nameX, $nameW, $namebMatmul)');
          break;
        case OP_LAYER_NORM_FORWARD:
          String nameInNorm = _readString();
          String nameGamma = _readString();
          String nameBetaNorm = _readString();
          String nameOutNorm = _readString();
          String nameMean = _readString();
          String nameRstd = _readString();
          double eps = _readFloat();
          Logger.cyan('$opName: $nameOutNorm = layer_norm($nameInNorm) eps:$eps');
          break;
        case OP_LAYER_NORM_BACKWARD:
          String gradOutNorm = _readString();
          String nameInNormBw = _readString();
          String nameGammaBw = _readString();
          String nameMeanBw = _readString();
          String nameRstdBw = _readString();
          String gradInNorm = _readString();
          String gradGamma = _readString();
          String gradBeta = _readString();
          Logger.blue('$opName: bw($gradOutNorm, in:$nameInNormBw)');
          break;
        case OP_CONV2D_FORWARD:
          String nameInConv = _readString();
          String nameKernel = _readString();
          String nameOutConv = _readString();
          Logger.cyan('$opName: $nameOutConv = conv2d($nameInConv, $nameKernel)');
          break;
        case OP_CONV2D_BACKWARD_INPUT:
          String gradOutConv = _readString();
          String nameKernelBw = _readString();
          String gradInConv = _readString();
          Logger.blue('$opName: $gradInConv = conv2d_bw_in($gradOutConv, $nameKernelBw)');
          break;
        case OP_CONV2D_BACKWARD_KERNEL:
          String nameInConvBw = _readString();
          String gradOutConvBw = _readString();
          String gradKernel = _readString();
          Logger.blue('$opName: $gradKernel = conv2d_bw_k($nameInConvBw, $gradOutConvBw)');
          break;
        case OP_CONV2D_MULTI_FORWARD:
          String nameInMulti = _readString();
          String nameWeight = _readString();
          String nameBiasMulti = _readString();
          String nameOutMulti = _readString();
          int inC = _readInt();
          int outC = _readInt();
          int kh = _readInt();
          int kw = _readInt();
          int pt = _readInt();
          int pl = _readInt();
          int sh = _readInt();
          int sw = _readInt();
          Logger.cyan('$opName: $nameOutMulti = conv2d_multi($nameInMulti, inC:$inC, outC:$outC) pad:${pt}x$pl stride:${sh}x$sw');
          break;
        case OP_CONV2D_MULTI_BACKWARD_INPUT:
          String gradOutMulti = _readString();
          String nameWeightMulti = _readString();
          String nameGradInMulti = _readString();
          int inCBw = _readInt();
          int outCBw = _readInt();
          int khBw = _readInt();
          int kwBw = _readInt();
          int ptBw = _readInt();
          int plBw = _readInt();
          int shBw = _readInt();
          int swBw = _readInt();
          Logger.blue('$opName: $nameGradInMulti = conv2d_multi_bw_in($gradOutMulti) inC:$inCBw, outC:$outCBw');
          break;
        case OP_CONV2D_MULTI_BACKWARD_WEIGHT:
          String nameInMultiBw = _readString();
          String gradOutMultiBw = _readString();
          String nameGradWeight = _readString();
          String nameGradBias = _readString();
          int inCW = _readInt();
          int outCW = _readInt();
          int khW = _readInt();
          int kwW = _readInt();
          int ptW = _readInt();
          int plW = _readInt();
          int shW = _readInt();
          int swW = _readInt();
          Logger.blue('$opName: $nameGradWeight, $nameGradBias = conv2d_multi_bw_w($nameInMultiBw, $gradOutMultiBw)');
          break;
        case OP_IM2COL:
          String nameInIm = _readString();
          String nameOutIm = _readString();
          int khIm = _readInt();
          int kwIm = _readInt();
          Logger.cyan('$opName: $nameOutIm = im2col($nameInIm) kernel: ${khIm}x$kwIm');
          break;
        case OP_COL2IM:
          String nameColGrad = _readString();
          String nameInGrad = _readString();
          int khCol = _readInt();
          int kwCol = _readInt();
          Logger.blue('$opName: $nameInGrad = col2im($nameColGrad) kernel: ${khCol}x$kwCol');
          break;
        case OP_MAX_POOL_1D_FORWARD:
        case OP_MAX_POOL_2D_FORWARD:
          String nameInPool = _readString();
          String nameOutPool = _readString();
          String nameIndicesPool = _readString();
          int poolSize = _readInt();
          int stridePool = _readInt();
          Logger.cyan('$opName: $nameOutPool = max_pool($nameInPool, size: $poolSize, stride: $stridePool)');
          break;
        case OP_MAX_POOL_1D_BACKWARD:
        case OP_MAX_POOL_2D_BACKWARD:
          String gradOutPool = _readString();
          String nameIndicesPoolBw = _readString();
          String gradInPool = _readString();
          Logger.blue('$opName: $gradInPool = max_pool_bw($gradOutPool)');
          break;
        case OP_AVG_POOL_2D_FORWARD: // Avg pool does NOT read indices in C++!
          String nameInAvg = _readString();
          String nameOutAvg = _readString();
          int avgPoolSize = _readInt();
          int avgStridePool = _readInt();
          Logger.cyan('$opName: $nameOutAvg = avg_pool($nameInAvg, size: $avgPoolSize, stride: $avgStridePool)');
          break;
        case OP_AVG_POOL_2D_BACKWARD: // Avg pool does NOT read indices in C++!
          String gradOutAvg = _readString();
          String gradInAvg = _readString();
          int avgPoolSizeBw = _readInt();
          int avgStridePoolBw = _readInt();
          Logger.blue('$opName: $gradInAvg = avg_pool_bw($gradOutAvg)');
          break;
        case OP_GLOBAL_AVG_POOL_FORWARD:
          String nameInGap = _readString();
          String nameOutGap = _readString();
          Logger.cyan('$opName: $nameOutGap = global_avg_pool($nameInGap)');
          break;
        case OP_GLOBAL_AVG_POOL_BACKWARD:
          String gradOutGap = _readString();
          String gradInGap = _readString();
          Logger.blue('$opName: $gradInGap = global_avg_pool_bw($gradOutGap)');
          break;
        case OP_BATCH_NORM_1D_FORWARD:
        case OP_BATCH_NORM_2D_FORWARD:
          String nameInBn = _readString();
          String nameGammaBn = _readString();
          String nameBetaBn = _readString();
          String nameRm = _readString();
          String nameRv = _readString();
          String nameOutBn = _readString();
          String nameSm = _readString();
          String nameSiv = _readString();
          double momentum = _readFloat();
          double epsilonBn = _readFloat();
          bool isTraining = _readBool();
          Logger.cyan('$opName: $nameOutBn = batch_norm($nameInBn) train:$isTraining');
          break;
        case OP_BATCH_NORM_1D_BACKWARD:
        case OP_BATCH_NORM_2D_BACKWARD:
          String gradOutBn = _readString();
          String nameInBnBw = _readString();
          String nameGammaBnBw = _readString();
          String nameSmBw = _readString();
          String nameSivBw = _readString();
          String gradInBn = _readString();
          String gradGammaBn = _readString();
          String gradBetaBn = _readString();
          Logger.blue('$opName: bw($gradOutBn, in:$nameInBnBw)');
          break;
        case OP_DROPOUT_FORWARD:
          String nameInDrop = _readString();
          String nameOutDrop = _readString();
          String nameMaskDrop = _readString();
          double dropRate = _readFloat();
          int seed = _readInt();
          Logger.cyan('$opName: $nameOutDrop = dropout($nameInDrop, rate: $dropRate, seed: $seed)');
          break;
        case OP_DROPOUT_BACKWARD:
          String gradOutDrop = _readString();
          String nameMaskDropBw = _readString();
          String gradInDrop = _readString();
          Logger.blue('$opName: $gradInDrop = dropout_bw($gradOutDrop, mask: $nameMaskDropBw)');
          break;

        default:
          Logger.red('Unknown OpCode ($opCode) encountered!', prefix: '⚠️');
          return;
      }
    }
    Logger.yellow('--- End of Tape ---', prefix: '📜');
  }

  // ─────────────────────────────────────────────────────── //
  // Internal Buffer Readers
  // ─────────────────────────────────────────────────────── //

  int _readInt() {
    int val = _reader.getInt32(_offset, Endian.host);
    _offset = _offset + 4;
    return val;
  }

  double _readFloat() {
    double val = _reader.getFloat32(_offset, Endian.host);
    _offset = _offset + 4;
    return val;
  }

  bool _readBool() {
    bool val = false;
    if (_reader.getInt8(_offset) == 1) {
      val = true;
    }
    _offset = _offset + 1;
    return val;
  }

  String _readString() {
    int len = _reader.getUint16(_offset, Endian.host);
    _offset = _offset + 2;

    Uint8List units = tape.sublist(_offset, _offset + len);
    String str = utf8.decode(units);
    _offset = _offset + len;
    return str;
  }

  String _getOpName(int code) {
    Map<int, String> map = {
      OP_LOAD_SAMPLE: 'OP_LOAD_SAMPLE',
      OP_STORE_SAMPLE: 'OP_STORE_SAMPLE',
      OP_COPY: 'OP_COPY',
      OP_FILL: 'OP_FILL',
      OP_ZERO_GRAD: 'OP_ZERO_GRAD',
      OP_ADD: 'OP_ADD',
      OP_ADD_INTO: 'OP_ADD_INTO',
      OP_SUBTRACT: 'OP_SUBTRACT',
      OP_SUBTRACT_INTO: 'OP_SUBTRACT_INTO',
      OP_MULTIPLY: 'OP_MULTIPLY',
      OP_MULTIPLY_BACKWARD: 'OP_MULTIPLY_BACKWARD',
      OP_DIVIDE: 'OP_DIVIDE',
      OP_DIVIDE_BACKWARD: 'OP_DIVIDE_BACKWARD',
      OP_ADD_SCALAR: 'OP_ADD_SCALAR',
      OP_LOG_ELEMENTWISE: 'OP_LOG_ELEMENTWISE',
      OP_LOG_BACKWARD: 'OP_LOG_BACKWARD',
      OP_ABS_ELEMENTWISE: 'OP_ABS_ELEMENTWISE',
      OP_ABS_BACKWARD: 'OP_ABS_BACKWARD',
      OP_SQRT_ELEMENTWISE: 'OP_SQRT_ELEMENTWISE',
      OP_SQRT_BACKWARD: 'OP_SQRT_BACKWARD',
      OP_POW_ELEMENTWISE: 'OP_POW_ELEMENTWISE',
      OP_POW_BACKWARD: 'OP_POW_BACKWARD',
      OP_CLAMP_ELEMENTWISE: 'OP_CLAMP_ELEMENTWISE',
      OP_CLAMP_BACKWARD: 'OP_CLAMP_BACKWARD',
      OP_MATMUL: 'OP_MATMUL',
      OP_TRANSPOSE: 'OP_TRANSPOSE',
      OP_BROADCAST_ADD: 'OP_BROADCAST_ADD',
      OP_SCALE_MATRIX: 'OP_SCALE_MATRIX',
      OP_SCALE_MATRIX_BACKWARD: 'OP_SCALE_MATRIX_BACKWARD',
      OP_RELU: 'OP_RELU',
      OP_RELU_BACKWARD: 'OP_RELU_BACKWARD',
      OP_SIGMOID: 'OP_SIGMOID',
      OP_SIGMOID_BACKWARD: 'OP_SIGMOID_BACKWARD',
      OP_TANH: 'OP_TANH',
      OP_TANH_BACKWARD: 'OP_TANH_BACKWARD',
      OP_EXP_ELEMENTWISE: 'OP_EXP_ELEMENTWISE',
      OP_EXP_BACKWARD: 'OP_EXP_BACKWARD',
      OP_SOFTMAX_FORWARD: 'OP_SOFTMAX_FORWARD',
      OP_SOFTMAX_BACKWARD: 'OP_SOFTMAX_BACKWARD',
      OP_GELU_FORWARD: 'OP_GELU_FORWARD',
      OP_GELU_BACKWARD: 'OP_GELU_BACKWARD',
      OP_MSE_LOSS_FORWARD: 'OP_MSE_LOSS_FORWARD',
      OP_MSE_LOSS_BACKWARD: 'OP_MSE_LOSS_BACKWARD',
      OP_BCE_LOSS_FORWARD: 'OP_BCE_LOSS_FORWARD',
      OP_BCE_LOSS_BACKWARD: 'OP_BCE_LOSS_BACKWARD',
      OP_SGD_UPDATE: 'OP_SGD_UPDATE',
      OP_ADAM_UPDATE: 'OP_ADAM_UPDATE',
      OP_CLIP_GRAD_VALUE: 'OP_CLIP_GRAD_VALUE',
      OP_SUM_REDUCE: 'OP_SUM_REDUCE',
      OP_SUM_REDUCE_BACKWARD: 'OP_SUM_REDUCE_BACKWARD',
      OP_SUM_REDUCE_COLUMNS: 'OP_SUM_REDUCE_COLUMNS',
      OP_SUM_REDUCE_ROWS: 'OP_SUM_REDUCE_ROWS',
      OP_EMBEDDING_FORWARD: 'OP_EMBEDDING_FORWARD',
      OP_EMBEDDING_BACKWARD: 'OP_EMBEDDING_BACKWARD',
      OP_SLICE_ROW: 'OP_SLICE_ROW',
      OP_SLICE_ROW_BACKWARD: 'OP_SLICE_ROW_BACKWARD',
      OP_SLICE_COLUMN: 'OP_SLICE_COLUMN',
      OP_SLICE_COLUMN_BACKWARD: 'OP_SLICE_COLUMN_BACKWARD',
      OP_STACK_ROWS: 'OP_STACK_ROWS',
      OP_STACK_ROWS_BACKWARD: 'OP_STACK_ROWS_BACKWARD',
      OP_CONCATENATE: 'OP_CONCATENATE',
      OP_CONCATENATE_BACKWARD: 'OP_CONCATENATE_BACKWARD',
      OP_PAD2D: 'OP_PAD2D',
      OP_PAD2D_BACKWARD: 'OP_PAD2D_BACKWARD',
      OP_MATMUL_BIAS_RELU_FORWARD: 'OP_MATMUL_BIAS_RELU_FORWARD',
      OP_LAYER_NORM_FORWARD: 'OP_LAYER_NORM_FORWARD',
      OP_LAYER_NORM_BACKWARD: 'OP_LAYER_NORM_BACKWARD',
      OP_CONV2D_FORWARD: 'OP_CONV2D_FORWARD',
      OP_CONV2D_BACKWARD_INPUT: 'OP_CONV2D_BACKWARD_INPUT',
      OP_CONV2D_BACKWARD_KERNEL: 'OP_CONV2D_BACKWARD_KERNEL',
      OP_CONV2D_MULTI_FORWARD: 'OP_CONV2D_MULTI_FORWARD',
      OP_CONV2D_MULTI_BACKWARD_INPUT: 'OP_CONV2D_MULTI_BACKWARD_INPUT',
      OP_CONV2D_MULTI_BACKWARD_WEIGHT: 'OP_CONV2D_MULTI_BACKWARD_WEIGHT',
      OP_IM2COL: 'OP_IM2COL',
      OP_COL2IM: 'OP_COL2IM',
      OP_MAX_POOL_1D_FORWARD: 'OP_MAX_POOL_1D_FORWARD',
      OP_MAX_POOL_1D_BACKWARD: 'OP_MAX_POOL_1D_BACKWARD',
      OP_MAX_POOL_2D_FORWARD: 'OP_MAX_POOL_2D_FORWARD',
      OP_MAX_POOL_2D_BACKWARD: 'OP_MAX_POOL_2D_BACKWARD',
      OP_AVG_POOL_2D_FORWARD: 'OP_AVG_POOL_2D_FORWARD',
      OP_AVG_POOL_2D_BACKWARD: 'OP_AVG_POOL_2D_BACKWARD',
      OP_GLOBAL_AVG_POOL_FORWARD: 'OP_GLOBAL_AVG_POOL_FORWARD',
      OP_GLOBAL_AVG_POOL_BACKWARD: 'OP_GLOBAL_AVG_POOL_BACKWARD',
      OP_BATCH_NORM_1D_FORWARD: 'OP_BATCH_NORM_1D_FORWARD',
      OP_BATCH_NORM_1D_BACKWARD: 'OP_BATCH_NORM_1D_BACKWARD',
      OP_BATCH_NORM_2D_FORWARD: 'OP_BATCH_NORM_2D_FORWARD',
      OP_BATCH_NORM_2D_BACKWARD: 'OP_BATCH_NORM_2D_BACKWARD',
      OP_DROPOUT_FORWARD: 'OP_DROPOUT_FORWARD',
      OP_DROPOUT_BACKWARD: 'OP_DROPOUT_BACKWARD'
    };

    if (map.containsKey(code)) {
      return map[code]!;
    }
    return 'UNKNOWN_OP_$code';
  }
}