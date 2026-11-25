import 'dart:convert';
import 'dart:typed_data';
import '/full_library.dart';
import 'dart_cpu_manager.dart';

class DartGraphExecutor {
  DartCpuManager manager;
  bool debug;

  late Uint8List _tape;
  late ByteData _reader;
  int _offset = 0;

  DartGraphExecutor(this.manager, {this.debug = false});

  void runTape(Uint8List tape) {
    _tape = tape;
    _reader = ByteData.sublistView(tape);
    _offset = 0;

    while (_offset < _tape.length) {
      int op = _readInt();

      switch (op) {
      // --- 0 - 99: Data & Memory Management ---
        case OP_LOAD_SAMPLE:
          String nameIn = _readString();
          String nameOut = _readString();
          int sampleIdx = _readInt();
          manager.sliceRow(nameIn, nameOut, sampleIdx);
          break;
        case OP_STORE_SAMPLE:
          String nameIn = _readString();
          String nameDest = _readString();
          int sampleIdx = _readInt();
          manager.sliceRowBackward(nameIn, nameDest, sampleIdx);
          break;
        case OP_COPY:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.copy(nameIn, nameOut);
          break;
        case OP_FILL:
          String nameOut = _readString();
          double value = _readFloat();
          manager.fill(nameOut, value);
          break;
        case OP_ZERO_GRAD:
          String name = _readString();
          manager.zeroGrad(name);
          break;

      // --- 100 - 199: Basic Math (Scalars & Element-wise) ---
        case OP_ADD:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          manager.add(nameA, nameB, nameC);
          break;
        case OP_ADD_INTO:
          String nameSrc = _readString();
          String nameDest = _readString();
          manager.addInto(nameSrc, nameDest);
          break;
        case OP_ADD_SCALAR:
          String nameIn = _readString();
          String nameOut = _readString();
          double scalar = _readFloat();
          manager.addScalar(nameIn, nameOut, scalar);
          break;
        case OP_SUBTRACT:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          manager.subtract(nameA, nameB, nameC);
          break;
        case OP_SUBTRACT_INTO:
          String nameSrc = _readString();
          String nameDest = _readString();
          manager.subtractInto(nameSrc, nameDest);
          break;
        case OP_MULTIPLY:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          manager.multiply(nameA, nameB, nameC);
          break;
        case OP_MULTIPLY_BACKWARD:
          String nameGradOut = _readString();
          String nameOtherIn = _readString();
          String nameGradIn = _readString();
          manager.multiplyBackward(nameGradOut, nameOtherIn, nameGradIn);
          break;
        case OP_DIVIDE:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          manager.divide(nameA, nameB, nameC);
          break;
        case OP_DIVIDE_BACKWARD:
          String nameA = _readString();
          String nameB = _readString();
          String nameGradOut = _readString();
          String nameGradA = _readString();
          String nameGradB = _readString();
          manager.divideBackward(
              nameA, nameB, nameGradOut, nameGradA, nameGradB);
          break;
        case OP_EXP_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.expElementwise(nameIn, nameOut);
          break;
        case OP_EXP_BACKWARD:
          String nameGradOut = _readString();
          String nameOutData = _readString();
          String nameGradIn = _readString();
          manager.expBackward(nameGradOut, nameOutData, nameGradIn);
          break;
        case OP_LOG_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.logElementwise(nameIn, nameOut);
          break;
        case OP_LOG_BACKWARD:
          String nameGradOut = _readString();
          String nameInData = _readString();
          String nameGradIn = _readString();
          manager.logBackward(nameGradOut, nameInData, nameGradIn);
          break;
        case OP_ABS_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.absElementwise(nameIn, nameOut);
          break;
        case OP_ABS_BACKWARD:
          String nameGradOut = _readString();
          String nameInData = _readString();
          String nameGradIn = _readString();
          manager.absBackward(nameGradOut, nameInData, nameGradIn);
          break;
        case OP_SQRT_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.sqrtElementwise(nameIn, nameOut);
          break;
        case OP_SQRT_BACKWARD:
          String nameGradOut = _readString();
          String nameOutData = _readString();
          String nameGradIn = _readString();
          manager.sqrtBackward(nameGradOut, nameOutData, nameGradIn);
          break;
        case OP_POW_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          double exponent = _readFloat();
          manager.powElementwise(nameIn, nameOut, exponent);
          break;
        case OP_POW_BACKWARD:
          String nameGradOut = _readString();
          String nameInData = _readString();
          String nameGradIn = _readString();
          double exponent = _readFloat();
          manager.powBackward(nameGradOut, nameInData, nameGradIn, exponent);
          break;
        case OP_CLAMP_ELEMENTWISE:
          String nameIn = _readString();
          String nameOut = _readString();
          double minVal = _readFloat();
          double maxVal = _readFloat();
          manager.clampElementwise(nameIn, nameOut, minVal, maxVal);
          break;
        case OP_CLAMP_BACKWARD:
          String nameGradOut = _readString();
          String nameInData = _readString();
          String nameGradIn = _readString();
          double minVal = _readFloat();
          double maxVal = _readFloat();
          manager.clampBackward(
              nameGradOut, nameInData, nameGradIn, minVal, maxVal);
          break;

      // --- 200 - 299: Matrix Operations ---
        case OP_MATMUL:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          bool transA = _readBool();
          bool transB = _readBool();
          double alpha = _readFloat();
          double beta = _readFloat();
          bool tCores = _readBool();
          manager.matmul(
              nameA, nameB, nameC, transA, transB, alpha, beta, tCores);
          break;
        case OP_TRANSPOSE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.transpose(nameIn, nameOut);
          break;
        case OP_BROADCAST_ADD:
          String nameA = _readString();
          String nameB = _readString();
          String nameC = _readString();
          manager.broadcastAdd(nameA, nameB, nameC);
          break;
        case OP_SCALE_MATRIX:
          String inName = _readString();
          String outName = _readString();
          double scale = _readFloat();
          manager.scaleMatrix(inName, outName, scale);
          break;
        case OP_SCALE_MATRIX_BACKWARD:
          String gradOutName = _readString();
          String gradInName = _readString();
          double scale = _readFloat();
          manager.scaleMatrixBackward(gradOutName, gradInName, scale);
          break;

      // --- 300 - 399: Activations ---
        case OP_RELU:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.relu(nameIn, nameOut);
          break;
        case OP_RELU_BACKWARD:
          String nameIn = _readString();
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          manager.reluBackward(nameIn, nameGradOut, nameGradIn);
          break;
        case OP_SIGMOID:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.sigmoid(nameIn, nameOut);
          break;
        case OP_SIGMOID_BACKWARD:
          String nameOutData = _readString();
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          manager.sigmoidBackward(nameOutData, nameGradOut, nameGradIn);
          break;
        case OP_TANH:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.tanh(nameIn, nameOut);
          break;
        case OP_TANH_BACKWARD:
          String nameOutData = _readString();
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          manager.tanhBackward(nameOutData, nameGradOut, nameGradIn);
          break;
        case OP_GELU_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.geluForward(nameIn, nameOut);
          break;
        case OP_GELU_BACKWARD:
          String nameIn = _readString();
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          manager.geluBackward(nameIn, nameGradOut, nameGradIn);
          break;
        case OP_SOFTMAX_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.softmaxForward(nameIn, nameOut);
          break;
        case OP_SOFTMAX_BACKWARD:
          String nameGradOut = _readString();
          String nameOutData = _readString();
          String nameGradIn = _readString();
          manager.softmaxBackward(nameGradOut, nameOutData, nameGradIn);
          break;

      // --- 400 - 499: Loss Functions ---
        case OP_MSE_LOSS_FORWARD:
          String namePred = _readString();
          String nameTarget = _readString();
          String nameOut = _readString();
          manager.mseLossForward(namePred, nameTarget, nameOut);
          break;
        case OP_MSE_LOSS_BACKWARD:
          String nameGradOut = _readString();
          String namePred = _readString();
          String nameTarget = _readString();
          String nameGradIn = _readString();
          manager.mseLossBackward(namePred, nameTarget, nameGradIn, nameGradOut);
          break;
        case OP_BCE_LOSS_FORWARD:
          String namePred = _readString();
          String nameTarget = _readString();
          String nameOut = _readString();
          manager.bceLossForward(namePred, nameTarget, nameOut);
          break;
        case OP_BCE_LOSS_BACKWARD:
          String nameGradOut = _readString();
          String namePred = _readString();
          String nameTarget = _readString();
          String nameGradIn = _readString();
          manager.bceLossBackward(nameGradOut, namePred, nameTarget, nameGradIn);
          break;

      // --- 500 - 599: Optimizers ---
        case OP_SGD_UPDATE:
          String nameData = _readString();
          String nameGrad = _readString();
          double lr = _readFloat();
          manager.sgdUpdate(nameData, nameGrad, lr);
          break;
        case OP_ADAM_UPDATE:
          String nameData = _readString();
          String nameGrad = _readString();
          String nameM = _readString();
          String nameV = _readString();
          double lr = _readFloat();
          double b1 = _readFloat();
          double b2 = _readFloat();
          double eps = _readFloat();
          int step = _readInt();
          double wd = _readFloat();
          manager.adamUpdate(
              nameData, nameGrad, nameM, nameV, lr, b1, b2, eps, step, wd);
          break;
        case OP_CLIP_GRAD_VALUE:
          String nameBuffer = _readString();
          double clipVal = _readFloat();
          manager.clipGradValue(nameBuffer, clipVal);
          break;

      // --- 600 - 699: Reductions ---
        case OP_SUM_REDUCE:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.sumReduce(nameIn, nameOut);
          break;
        case OP_SUM_REDUCE_BACKWARD:
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          manager.sumReduceBackward(nameGradOut, nameGradIn);
          break;
        case OP_SUM_REDUCE_COLUMNS:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.sumReduceColumns(nameIn, nameOut);
          break;
        case OP_SUM_REDUCE_ROWS:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.sumReduceRows(nameIn, nameOut);
          break;
        case OP_EMBEDDING_FORWARD:
          String nameIndices = _readString();
          String nameWeight = _readString();
          String nameOut = _readString();
          manager.embeddingForward(nameIndices, nameWeight, nameOut);
          break;
        case OP_EMBEDDING_BACKWARD:
          String nameGradOut = _readString();
          String nameIndices = _readString();
          String nameGradWeight = _readString();
          manager.embeddingBackward(nameGradOut, nameIndices, nameGradWeight);
          break;

      // --- 700 - 799: Tensor Manipulation ---
        case OP_SLICE_ROW:
          String nameIn = _readString();
          String nameOut = _readString();
          int row = _readInt();
          manager.sliceRow(nameIn, nameOut, row);
          break;
        case OP_SLICE_ROW_BACKWARD:
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          int row = _readInt();
          manager.sliceRowBackward(nameGradOut, nameGradIn, row);
          break;
        case OP_SLICE_COLUMN:
          String nameIn = _readString();
          String nameOut = _readString();
          int start = _readInt();
          int end = _readInt();
          manager.sliceColumn(nameIn, nameOut, start, end);
          break;
        case OP_SLICE_COLUMN_BACKWARD:
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          int start = _readInt();
          int end = _readInt();
          manager.sliceColumnBackward(nameGradOut, nameGradIn, start, end);
          break;
        case OP_STACK_ROWS:
          int count = _readInt();
          List<String> namesIn = <String>[];
          for (int i = 0; i < count; i = i + 1) {
            namesIn.add(_readString());
          }
          String nameOut = _readString();
          int axis = _readInt();
          manager.stackRows(namesIn, nameOut, axis);
          break;
        case OP_STACK_ROWS_BACKWARD:
          String nameGradOut = _readString();
          int count = _readInt();
          List<String> namesGradIn = <String>[];
          for (int i = 0; i < count; i = i + 1) {
            namesGradIn.add(_readString());
          }
          int axis = _readInt();
          manager.stackRowsBackward(nameGradOut, namesGradIn, axis);
          break;
        case OP_CONCATENATE:
          String nameA = _readString();
          String nameB = _readString();
          String nameOut = _readString();
          int axis = _readInt();
          manager.concatenate(nameA, nameB, nameOut, axis);
          break;
        case OP_CONCATENATE_BACKWARD:
          String nameGradOut = _readString();
          String nameGradInA = _readString();
          String nameGradInB = _readString();
          int axis = _readInt();
          int split = _readInt();
          manager.concatenateBackward(
              nameGradOut, nameGradInA, nameGradInB, axis, split);
          break;
        case OP_PAD2D:
          String nameIn = _readString();
          String nameOut = _readString();
          int padT = _readInt();
          int padB = _readInt();
          int padL = _readInt();
          int padR = _readInt();
          manager.pad2d(nameIn, nameOut, padT, padB, padL, padR);
          break;
        case OP_PAD2D_BACKWARD:
          String nameGradOut = _readString();
          String nameGradIn = _readString();
          int padT = _readInt();
          int padB = _readInt();
          int padL = _readInt();
          int padR = _readInt();
          manager.pad2dBackward(nameGradOut, nameGradIn, padT, padB, padL, padR);
          break;

      // --- 800 - 999: Advanced Spatial & Sequence Layers ---
        case OP_CONV2D_FORWARD:
          String nameIn = _readString();
          String nameKernel = _readString();
          String nameOut = _readString();
          manager.conv2dForward(nameIn, nameKernel, nameOut);
          break;
        case OP_CONV2D_BACKWARD_INPUT:
          String nameGradOut = _readString();
          String nameKernel = _readString();
          String nameGradIn = _readString();
          manager.conv2dBackwardInput(nameGradOut, nameKernel, nameGradIn);
          break;
        case OP_CONV2D_BACKWARD_KERNEL:
          String nameIn = _readString();
          String nameGradOut = _readString();
          String nameGradKernel = _readString();
          manager.conv2dBackwardKernel(nameIn, nameGradOut, nameGradKernel);
          break;
        case OP_CONV2D_MULTI_FORWARD:
          String nameIn = _readString();
          String nameW = _readString();
          String nameB = _readString();
          String nameOut = _readString();
          int cIn = _readInt();
          int cOut = _readInt();
          int kH = _readInt();
          int kW = _readInt();
          int padT = _readInt();
          int padL = _readInt();
          int strideH = _readInt();
          int strideW = _readInt();
          manager.conv2dMultiForward(nameIn, nameW, nameB, nameOut, cIn, cOut,
              kH, kW, padT, padL, strideH, strideW);
          break;
        case OP_CONV2D_MULTI_BACKWARD_INPUT:
          String nameGradOut = _readString();
          String nameWeight = _readString();
          String nameGradIn = _readString();
          int cIn = _readInt();
          int cOut = _readInt();
          int kH = _readInt();
          int kW = _readInt();
          int padT = _readInt();
          int padL = _readInt();
          int strideH = _readInt();
          int strideW = _readInt();
          manager.conv2dMultiBackwardInput(nameGradOut, nameWeight, nameGradIn,
              cIn, cOut, kH, kW, padT, padL, strideH, strideW);
          break;
        case OP_CONV2D_MULTI_BACKWARD_WEIGHT:
          String nameInput = _readString();
          String nameGradOut = _readString();
          String nameGradWeight = _readString();
          String nameGradBias = _readString();
          int cIn = _readInt();
          int cOut = _readInt();
          int kH = _readInt();
          int kW = _readInt();
          int padT = _readInt();
          int padL = _readInt();
          int strideH = _readInt();
          int strideW = _readInt();
          manager.conv2dMultiBackwardWeight(
              nameInput,
              nameGradOut,
              nameGradWeight,
              nameGradBias,
              cIn,
              cOut,
              kH,
              kW,
              padT,
              padL,
              strideH,
              strideW);
          break;
        case OP_IM2COL:
          String nameIn = _readString();
          String nameOut = _readString();
          int kh = _readInt();
          int kw = _readInt();
          manager.im2col(nameIn, nameOut, kh, kw);
          break;
        case OP_COL2IM:
          String nameColGrad = _readString();
          String nameInGrad = _readString();
          int kh = _readInt();
          int kw = _readInt();
          manager.col2im(nameColGrad, nameInGrad, kh, kw);
          break;
        case OP_MAX_POOL_1D_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          String nameIndices = _readString();
          int poolSize = _readInt();
          int stride = _readInt();
          manager.maxPool1dForward(
              nameIn, nameOut, nameIndices, poolSize, stride);
          break;
        case OP_MAX_POOL_1D_BACKWARD:
          String nameGradOut = _readString();
          String nameIndices = _readString();
          String nameGradIn = _readString();
          manager.maxPool1dBackward(nameGradOut, nameIndices, nameGradIn);
          break;
        case OP_MAX_POOL_2D_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          String nameIndices = _readString();
          int poolSize = _readInt();
          int stride = _readInt();
          manager.maxPool2dForward(
              nameIn, nameOut, nameIndices, poolSize, stride);
          break;
        case OP_MAX_POOL_2D_BACKWARD:
          String nameGradOut = _readString();
          String nameIndices = _readString();
          String nameGradIn = _readString();
          manager.maxPool2dBackward(nameGradOut, nameIndices, nameGradIn);
          break;
        case OP_AVG_POOL_2D_FORWARD:
          String inName = _readString();
          String outName = _readString();
          int poolSize = _readInt();
          int stride = _readInt();
          manager.avgPool2dForward(inName, outName, poolSize, stride);
          break;
        case OP_AVG_POOL_2D_BACKWARD:
          String gradOutName = _readString();
          String gradInName = _readString();
          int poolSize = _readInt();
          int stride = _readInt();
          manager.avgPool2dBackward(gradOutName, gradInName, poolSize, stride);
          break;
        case OP_GLOBAL_AVG_POOL_FORWARD:
          String inName = _readString();
          String outName = _readString();
          manager.globalAvgPoolForward(inName, outName);
          break;
        case OP_GLOBAL_AVG_POOL_BACKWARD:
          String gradOutName = _readString();
          String gradInName = _readString();
          manager.globalAvgPoolBackward(gradOutName, gradInName);
          break;
        case OP_BATCH_NORM_1D_FORWARD:
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameBeta = _readString();
          String nameRm = _readString();
          String nameRv = _readString();
          String nameOut = _readString();
          String nameSm = _readString();
          String nameSiv = _readString();
          double momentum = _readFloat();
          double epsilon = _readFloat();
          bool isTraining = _readBool();
          manager.batchNorm1dForward(nameIn, nameGamma, nameBeta, nameRm,
              nameRv, nameOut, nameSm, nameSiv, momentum, epsilon, isTraining);
          break;
        case OP_BATCH_NORM_1D_BACKWARD:
          String nameGradOut = _readString();
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameSm = _readString();
          String nameSiv = _readString();
          String nameGradIn = _readString();
          String nameGradGamma = _readString();
          String nameGradBeta = _readString();
          manager.batchNorm1dBackward(nameGradOut, nameIn, nameGamma, nameSm,
              nameSiv, nameGradIn, nameGradGamma, nameGradBeta);
          break;
        case OP_BATCH_NORM_2D_FORWARD:
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameBeta = _readString();
          String nameRm = _readString();
          String nameRv = _readString();
          String nameOut = _readString();
          String nameSm = _readString();
          String nameSiv = _readString();
          double momentum = _readFloat();
          double epsilon = _readFloat();
          bool isTraining = _readBool();
          manager.batchNorm2dForward(nameIn, nameGamma, nameBeta, nameRm,
              nameRv, nameOut, nameSm, nameSiv, momentum, epsilon, isTraining);
          break;
        case OP_BATCH_NORM_2D_BACKWARD:
          String nameGradOut = _readString();
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameSm = _readString();
          String nameSiv = _readString();
          String nameGradIn = _readString();
          String nameGradGamma = _readString();
          String nameGradBeta = _readString();
          manager.batchNorm2dBackward(nameGradOut, nameIn, nameGamma, nameSm,
              nameSiv, nameGradIn, nameGradGamma, nameGradBeta);
          break;
        case OP_LAYER_NORM_FORWARD:
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameBeta = _readString();
          String nameOut = _readString();
          String nameMean = _readString();
          String nameRstd = _readString();
          double eps = _readFloat();
          manager.layerNormForward(
              nameIn, nameGamma, nameBeta, nameOut, nameMean, nameRstd, eps);
          break;
        case OP_LAYER_NORM_BACKWARD:
          String nameGradOut = _readString();
          String nameIn = _readString();
          String nameGamma = _readString();
          String nameMean = _readString();
          String nameRstd = _readString();
          String nameGradIn = _readString();
          String nameGradGamma = _readString();
          String nameGradBeta = _readString();
          manager.layerNormBackward(nameGradOut, nameIn, nameGamma, nameMean,
              nameRstd, nameGradIn, nameGradGamma, nameGradBeta);
          break;
        case OP_DROPOUT_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          String nameMask = _readString();
          double dropRate = _readFloat();
          int seed = _readInt();
          manager.dropoutForward(nameIn, nameOut, nameMask, dropRate, seed);
          break;
        case OP_DROPOUT_BACKWARD:
          String nameGradOut = _readString();
          String nameMask = _readString();
          String nameGradIn = _readString();
          manager.dropoutBackward(nameGradOut, nameMask, nameGradIn);
          break;
        case OP_MARKOV_COUNT:
          String nameSeq = _readString();
          String nameCount = _readString();
          int order = _readInt();
          int numStates = _readInt();
          manager.markovCount(nameSeq, nameCount, order, numStates);
          break;
        case OP_MARKOV_NORMALIZE:
          String nameCount = _readString();
          String nameProb = _readString();
          int numHistories = _readInt();
          int numStates = _readInt();
          manager.markovNormalize(nameCount, nameProb, numHistories, numStates);
          break;
        case OP_MARKOV_PREDICT:
          String nameHist = _readString();
          String nameProb = _readString();
          String nameOut = _readString();
          int order = _readInt();
          int numStates = _readInt();
          manager.markovPredict(nameHist, nameProb, nameOut, order, numStates);
          break;

      // --- 1000+: Fused Kernels ---
        case OP_MATMUL_BIAS_RELU_FORWARD:
          String nameX = _readString();
          String nameW = _readString();
          String nameB = _readString();
          String nameReluOut = _readString();
          String namePreROut = _readString();
          manager.matmulBiasReluForward(
              nameX, nameW, nameB, nameReluOut, namePreROut);
          break;

      // --- 2000+: Transformer Ops ---
        case OP_RMS_NORM_FORWARD:
          String nameIn = _readString();
          String nameW = _readString();
          String nameOut = _readString();
          String nameRstd = _readString();
          double eps = _readFloat();
          bool scalePlusOne = _readBool();
          manager.rmsNormForward(
              nameIn, nameW, nameOut, nameRstd, eps, scalePlusOne);
          break;
        case OP_CAUSAL_MASK_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          int batch = _readInt();
          int heads = _readInt();
          int seq = _readInt();
          manager.causalMaskForward(nameIn, nameOut, batch, heads, seq);
          break;
        case OP_ROPE_FORWARD:
          String nameIn = _readString();
          String nameCos = _readString();
          String nameSin = _readString();
          String nameOut = _readString();
          int batch = _readInt();
          int seq = _readInt();
          int heads = _readInt();
          int dim = _readInt();
          manager.ropeForward(
              nameIn, nameCos, nameSin, nameOut, batch, seq, heads, dim);
          break;
        case OP_CROSS_ENTROPY_FORWARD:
          String nameLogits = _readString();
          String nameTargets = _readString();
          String nameOut = _readString();
          manager.crossEntropyForward(nameLogits, nameTargets, nameOut);
          break;
        case OP_ARGMAX_FORWARD:
          String nameIn = _readString();
          String nameOut = _readString();
          manager.argmaxForward(nameIn, nameOut);
          break;

        default:
          throw Exception('DartGraphExecutor: Unknown OpCode ($op)');
      }
    }
  }

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

    Uint8List units = _tape.sublist(_offset, _offset + len);
    String str = utf8.decode(units);
    _offset = _offset + len;
    return str;
  }
}