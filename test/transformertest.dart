import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

List<double> pullRawTensor(String id, int elementCount) {
  List<double> result = List<double>.filled(elementCount, 0.0);
  using((Arena arena) {
    Pointer<Float> ptr = arena<Float>(elementCount);
    GPUEngine.retrieve(id, ptr);
    Float32List view = ptr.asTypedList(elementCount);
    for (int i = 0; i < elementCount; i = i + 1) {
      result[i] = view[i];
    }
  });
  return result;
}

void main() {
  test('Pure-tape step-by-step diagnostic', () async {
    await GPUEngine.initialize(target: Target.cuda);

    // =========================================================================
    // PHASE 1: Compile Graph and Extract Tapes
    // =========================================================================
    int vocabSize = 10;
    int dModel = 16;
    int numHeads = 4;
    int dff = 32;
    int maxSeqLength = 8;

    AllocBuffer dataAlloc = AllocBuffer();

    GPUTensor<Vector> trainInput =
    GPUTensor<Vector>(<double>[1.0, 3.0, 5.0, 2.0], dataAlloc);

    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dataAlloc);

    DenseTL outputDense = DenseTL(vocabSize);

    List<TapeLayer> layers = <TapeLayer>[
      EmbeddingTL(vocabSize, dModel),
      PositionalEncodingTL(maxSeqLength, dModel),
      TransformerEncoderBlockTapeLayer(dModel, numHeads, dff),
      outputDense,
    ];

    SeqModel<Vector, Matrix> model = SeqModel<Vector, Matrix>(
      layers,
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params, AllocBuffer allocTape) {
        return SGDGPU(params, 0.01);
      },
    );

    model.compile();

    AllocBuffer modelAlloc = model.allocTape;
    Uint8List zeroGradTape = model.trainZeroGrad.bytes();
    Uint8List forwardTape = model.trainForward.bytes();
    Uint8List backwardTape = model.trainBackward.bytes();
    Uint8List optimizeTape = model.trainOptimize.bytes();

    String lossId = model.loss!.id;
    String trackedGradId = outputDense.weights.gradId;
    int trackedGradElements =
        outputDense.weights.shape[0] * outputDense.weights.shape[1];

    GPUEngine.dispose();

    // =========================================================================
    // PHASE 2: Step-by-Step Diagnostic Run
    // =========================================================================
    await GPUEngine.initialize(target: Target.cuda);

    dataAlloc.execute();
    modelAlloc.execute();

    List<double> gradAfterAlloc =
    pullRawTensor(trackedGradId, trackedGradElements).sublist(0, 4);
    Logger.blue('$trackedGradId right after alloc: $gradAfterAlloc');

    int epochs = 250;
    for (int epoch = 0; epoch < epochs; epoch = epoch + 1) {
      GPUEngine.run(zeroGradTape);
      List<double> gradAfterZero =
      pullRawTensor(trackedGradId, trackedGradElements).sublist(0, 4);

      GPUEngine.run(forwardTape);
      double lossAfterForward = pullRawTensor(lossId, 1)[0];

      GPUEngine.run(backwardTape);
      List<double> gradAfterBackward =
      pullRawTensor(trackedGradId, trackedGradElements).sublist(0, 4);

      GPUEngine.run(optimizeTape);

      Logger.blue(
        'Step $epoch | Forward Loss: $lossAfterForward | gradAfterZero: $gradAfterZero | gradAfterBw: $gradAfterBackward',
      );

      if (lossAfterForward.isNaN || lossAfterForward.isInfinite) {
        Logger.red('Aborted at Step $epoch due to non-finite loss.');
        break;
      }
    }

    modelAlloc.freeAll();
    dataAlloc.freeAll();
    GPUEngine.dispose();
  });
}