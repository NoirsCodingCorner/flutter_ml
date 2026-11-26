import 'dart:typed_data';

import 'full_library.dart';

void fillTensorGPU(GPUTensor tensor, double value, CommandBuffer tape) {
  tape.putInt(OP_FILL);
  tape.putString(tensor.id);
  tape.putFloat(value);
}

void runBenchmarkIsolated({
  required String name,
  required int iterations,
  required int bytesPerOp,
  required int flopsPerOp,
  required void Function(AllocBuffer allocTape) allocateInputs,
  required void Function(AllocBuffer allocTape) initData,
  required CommandBuffer Function(AllocBuffer allocTape) buildTape,
}) {
  AllocBuffer inputAlloc = AllocBuffer();
  AllocBuffer outputAlloc = AllocBuffer();
  Stopwatch allocSw = Stopwatch();

  // 1. Time Allocation of inputs
  allocSw.start();
  allocateInputs(inputAlloc);
  inputAlloc.execute();
  allocSw.stop();

  // 2. Initialize data (Excluded from overhead)
  AllocBuffer initAlloc = AllocBuffer();
  initData(initAlloc);
  initAlloc.freeAll();

  // 3. Time Allocation of outputs and tape generation
  allocSw.start();
  CommandBuffer tape = buildTape(outputAlloc);
  outputAlloc.execute();
  allocSw.stop();

  Uint8List tapeBytes = tape.bytes();

  // Warmup Engine
  for (int i = 0; i < 3; i = i + 1) {
    GPUEngine.run(tapeBytes);
  }

  // 4. Time Execution
  Stopwatch execSw = Stopwatch();
  execSw.start();
  for (int i = 0; i < iterations; i = i + 1) {
    GPUEngine.run(tapeBytes);
  }
  execSw.stop();

  // 5. Time Deallocation
  Stopwatch freeSw = Stopwatch();
  freeSw.start();
  outputAlloc.freeAll();
  inputAlloc.freeAll();
  freeSw.stop();

  // Calculations
  double seconds = execSw.elapsedMicroseconds / 1000000.0;
  double avgMs = (seconds * 1000.0) / iterations;
  double gbPerSec = (bytesPerOp * iterations / 1000000000.0) / seconds;
  double gflops = (flopsPerOp * iterations / 1000000000.0) / seconds;

  String timeStr = avgMs.toStringAsFixed(2).padLeft(6);
  String gbStr = gbPerSec.toStringAsFixed(2).padLeft(8);
  String gflopsStr = gflops.toStringAsFixed(4).padLeft(8);
  String allocStr =
  (allocSw.elapsedMicroseconds / 1000.0).toStringAsFixed(2).padLeft(6);
  String freeStr =
  (freeSw.elapsedMicroseconds / 1000.0).toStringAsFixed(2).padLeft(6);

  print(
      "[BENCHMARK] ${name.padRight(10)} | Time: $timeStr ms | Bandwidth: $gbStr GB/s | Compute: $gflopsStr GFLOPs             Overhead  | Alloc/Tape: $allocStr ms | Free: $freeStr ms");
}

void testAll(Target target) async {
  await GPUEngine.initialize(debug: false, target: target);

  int N = 2048 * 2048 * 8;
  List<int> vecShape = <int>[N];

  int M = 1024;
  List<int> matShape = <int>[M, M];

  int iterations = 50;

  print("\n==================================================================");
  print("                 CUDA ENGINE PERFORMANCE BENCHMARK                ");
  print("==================================================================");
  print(
      "Vector Size: $N elements (~${(N * 4 / 1000000).toStringAsFixed(0)} MB)");
  print("Matrix Size: ${M}x$M elements");
  print("Iterations:  $iterations");
  print(
      "Note: VRAM is aggressively wiped and reallocated between each run to test loading speeds of different operations. ");
  print("------------------------------------------------------------------\n");

  int unaryBytes = N * 8;
  int binaryBytes = N * 12;
  int elementFlops = N;
  int matmulBytes = M * M * 12;
  int matmulFlops = 2 * M * M * M;

  // --- BINARY OPERATIONS ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "ADD",
      iterations: iterations,
      bytesPerOp: binaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
        vecB = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.0, t);
        fillTensorGPU(vecB!, 3.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        addGPU<Vector>(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int markovSeqLen = 67108864; // ~67 Million elements
    int numStates = 16;
    int order = 2;

    int buildBytes = markovSeqLen * 4;
    int buildFlops = markovSeqLen;

    GPUTensor<Vector>? sequence;

    runBenchmarkIsolated(
      name: "MARKOV_TBL",
      iterations: iterations,
      bytesPerOp: buildBytes,
      flopsPerOp: buildFlops,
      allocateInputs: (AllocBuffer allocTape) {
        sequence = GPUTensor<Vector>.randomUniform(
            <int>[markovSeqLen], numStates / 2.0, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();

        GPUTensor<Vector> shifted =
        addScalarVectorGPU(sequence!, numStates / 2.0, t, allocTape);

        GPUTensor<Vector> clamped = clampGPU<Vector>(
            shifted, 0.0, (numStates - 1).toDouble(), t, allocTape);

        t.putInt(OP_COPY);
        t.putString(clamped.id);
        t.putString(sequence!.id);

        allocTape.execute();
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        buildMarkovTableGPU(sequence!, order, numStates, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int numStates = 16;
    int order = 2;
    int numHistories = 256; // 16^2

    int predictBatchSize = 10000000;

    int predictBytes =
        (predictBatchSize * order * 4) + (predictBatchSize * numStates * 4);
    int predictFlops = predictBatchSize * numStates;

    GPUTensor<Matrix>? historyBatch;
    GPUTensor<Matrix>? probTable;

    runBenchmarkIsolated(
      name: "MARKOV_PRD",
      iterations: iterations,
      bytesPerOp: predictBytes,
      flopsPerOp: predictFlops,
      allocateInputs: (AllocBuffer allocTape) {
        historyBatch =
        GPUTensor<Matrix>.empty(<int>[predictBatchSize, order], allocTape);
        probTable =
        GPUTensor<Matrix>.empty(<int>[numHistories, numStates], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(historyBatch!, 1.0, t);
        fillTensorGPU(probTable!, 0.0625, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        markovPredictGPU(historyBatch!, probTable!, numStates, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "SUBTRACT",
      iterations: iterations,
      bytesPerOp: binaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
        vecB = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 5.0, t);
        fillTensorGPU(vecB!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        subtractGPU<Vector>(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "MULTIPLY",
      iterations: iterations,
      bytesPerOp: binaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
        vecB = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.0, t);
        fillTensorGPU(vecB!, 3.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        multiplyGPU<Vector>(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "DIVIDE",
      iterations: iterations,
      bytesPerOp: binaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
        vecB = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 10.0, t);
        fillTensorGPU(vecB!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        divideGPU<Vector>(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  // --- UNARY OPERATIONS ---
      {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "ABS",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, -2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        absGPU<Vector>(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "SQRT",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 4.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sqrtGPU<Vector>(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "LOG",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.718, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        logGPU<Vector>(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "POW",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 3.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        powGPU<Vector>(vecA!, 3.0, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "CLAMP",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 10.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        clampGPU<Vector>(vecA!, 0.0, 5.0, tape, allocTape);
        return tape;
      },
    );
  }

  // --- MATMUL ---
      {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? matB;

    runBenchmarkIsolated(
      name: "MATMUL",
      iterations: iterations,
      bytesPerOp: matmulBytes,
      flopsPerOp: matmulFlops,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(matShape, allocTape);
        matB = GPUTensor<Matrix>.empty(matShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        fillTensorGPU(matB!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        GPUTensor<Matrix> outMatmul =
        GPUTensor<Matrix>.empty(matShape, allocTape);

        tape.putInt(OP_MATMUL);
        tape.putString(matA!.id);
        tape.putString(matB!.id);
        tape.putString(outMatmul.id);
        tape.putBool(false);
        tape.putBool(false);
        tape.putFloat(1.0);
        tape.putFloat(0.0);
        tape.putBool(true);

        return tape;
      },
    );
  }

  // --- LINEAR ALGEBRA & BROADCASTING ---
      {
    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "TRANSPOSE",
      iterations: iterations,
      bytesPerOp: M * M * 8,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        transposeGPU(matA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "MAT_VEC",
      iterations: iterations,
      bytesPerOp: (M * M * 4) + (M * 4) + (M * 4),
      flopsPerOp: 2 * M * M,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        fillTensorGPU(vecB!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        matVecMulGPU(matA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? vecBias;

    runBenchmarkIsolated(
      name: "ADD_BIAS",
      iterations: iterations,
      bytesPerOp: (M * M * 4) + (M * 4) + (M * M * 4),
      flopsPerOp: M * M,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
        vecBias = GPUTensor<Vector>.empty(<int>[M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        fillTensorGPU(vecBias!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        addBiasToMatMulOutGPU(matA!, vecBias!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SCALE_MAT",
      iterations: iterations,
      bytesPerOp: M * M * 8,
      flopsPerOp: M * M,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        scaleMatrixGPU(matA!, 3.0, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "ADD_SCALAR",
      iterations: iterations,
      bytesPerOp: N * 8,
      flopsPerOp: N,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        addScalarVectorGPU(vecA!, 5.0, tape, allocTape);
        return tape;
      },
    );
  }

  // --- ACTIVATIONS ---
      {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "RELU",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        reluGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "SIGMOID",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops * 3,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 0.5, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sigmoidGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "TANH",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops * 3,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 0.5, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        vectorTanhGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "GELU",
      iterations: iterations,
      bytesPerOp: unaryBytes,
      flopsPerOp: elementFlops * 5,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(vecShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 0.5, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        geluGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int softmaxBytes = M * M * 12;
    int softmaxFlops = M * M * 3;

    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SOFTMAX",
      iterations: iterations,
      bytesPerOp: softmaxBytes,
      flopsPerOp: softmaxFlops,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(matShape, allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        softmaxMatrixGPU(matA!, tape, allocTape);
        return tape;
      },
    );
  }

  // --- LOSS FUNCTIONS (Memory Bound) ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "BCE_LOSS",
      iterations: iterations,
      bytesPerOp: N * 8,
      flopsPerOp: N * 4,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 0.5, t);
        fillTensorGPU(vecB!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        binaryCrossEntropyGPU<Vector>(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "MSE_VEC",
      iterations: iterations,
      bytesPerOp: N * 8,
      flopsPerOp: N * 3,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.0, t);
        fillTensorGPU(vecB!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        mseGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? matB;

    runBenchmarkIsolated(
      name: "MSE_MAT",
      iterations: iterations,
      bytesPerOp: M * M * 8,
      flopsPerOp: M * M * 3,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
        matB = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 2.0, t);
        fillTensorGPU(matB!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        mseMatrixGPU(matA!, matB!, tape, allocTape);
        return tape;
      },
    );
  }

  // --- REDUCTIONS (Memory Bound) ---
      {
    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "SUM_VEC",
      iterations: iterations,
      bytesPerOp: N * 4,
      flopsPerOp: N,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sumGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SUM_COLS",
      iterations: iterations,
      bytesPerOp: (M * M * 4) + (M * 4),
      flopsPerOp: M * M,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sumReduceColumnsGPU(matA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SUM_ROWS",
      iterations: iterations,
      bytesPerOp: (M * M * 4) + (M * 4),
      flopsPerOp: M * M,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sumReduceRowsGPU(matA!, tape, allocTape);
        return tape;
      },
    );
  }

  // --- EMBEDDING LOOKUPS (Memory Scatter/Gather Bound) ---
      {
    int vocabSize = 32000;
    int embedDim = 768;
    int numTokens = 1048576;

    int embedBytes = (numTokens * 4) +
        (numTokens * embedDim * 4) +
        (numTokens * embedDim * 4);
    int embedFlops = 0;

    GPUTensor<Vector>? indices;
    GPUTensor<Matrix>? weights;

    runBenchmarkIsolated(
      name: "EMBED_VEC",
      iterations: iterations,
      bytesPerOp: embedBytes,
      flopsPerOp: embedFlops,
      allocateInputs: (AllocBuffer allocTape) {
        indices = GPUTensor<Vector>.empty(<int>[numTokens], allocTape);
        weights =
        GPUTensor<Matrix>.empty(<int>[vocabSize, embedDim], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(indices!, 0.0, t);
        fillTensorGPU(weights!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        embeddingLookupGPU(indices!, weights!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int vocabSize = 32000;
    int embedDim = 768;
    int batchSize = 1024;
    int seqLength = 1024;

    int embedBytes = (batchSize * seqLength * 4) +
        (batchSize * seqLength * embedDim * 4) +
        (batchSize * seqLength * embedDim * 4);

    GPUTensor<Matrix>? batchIndices;
    GPUTensor<Matrix>? weights;

    runBenchmarkIsolated(
      name: "EMBED_MAT",
      iterations: iterations,
      bytesPerOp: embedBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        batchIndices =
        GPUTensor<Matrix>.empty(<int>[batchSize, seqLength], allocTape);
        weights =
        GPUTensor<Matrix>.empty(<int>[vocabSize, embedDim], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(batchIndices!, 0.0, t);
        fillTensorGPU(weights!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        embeddingLookupBatchGPU(batchIndices!, weights!, tape, allocTape);
        return tape;
      },
    );
  }

  // --- TENSOR MANIPULATION & ROUTING (Memory Bound) ---
      {
    int sliceRows = M;
    int sliceCols = M ~/ 2;
    int sliceBytes = (sliceRows * sliceCols * 4) * 2;

    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SLICE_COL",
      iterations: iterations,
      bytesPerOp: sliceBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        sliceColumnGPU(matA!, 0, sliceCols, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int wideCols = 1048576;
    int rowBytes = wideCols * 4 * 2;

    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "SLICE_ROW",
      iterations: iterations,
      bytesPerOp: rowBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[128, wideCols], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        selectRowGPU(matA!, 64, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int depth = 64;
    int size = 1024;
    int slice3DBytes = (size * size * 4) * 2;

    GPUTensor<Tensor3D>? t3D;

    runBenchmarkIsolated(
      name: "SLICE_3D",
      iterations: iterations,
      bytesPerOp: slice3DBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        t3D = GPUTensor<Tensor3D>.empty(<int>[depth, size, size], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(t3D!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        selectMatrixFrom3DGPU(t3D!, 32, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int halfN = N ~/ 2;
    int concatBytes = N * 4 * 2;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "CONCAT_VEC",
      iterations: iterations,
      bytesPerOp: concatBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[halfN], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[halfN], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 1.0, t);
        fillTensorGPU(vecB!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        concatenateGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int numMatrices = 16;
    int rows = 1024;
    int cols = 1024;
    int stackBytes = (numMatrices * rows * cols * 4) * 2;

    List<GPUTensor<Matrix>>? matrices;

    runBenchmarkIsolated(
      name: "STACK_MAT",
      iterations: iterations,
      bytesPerOp: stackBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        matrices = <GPUTensor<Matrix>>[];
        for (int i = 0; i < numMatrices; i = i + 1) {
          matrices!.add(GPUTensor<Matrix>.empty(<int>[rows, cols], allocTape));
        }
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        for (int i = 0; i < matrices!.length; i = i + 1) {
          fillTensorGPU(matrices![i], 1.0, t);
        }
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        stackMatricesGPU(matrices!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int seqLen = 4096;
    int dHead = 64;
    int numHeads = 12;
    int dModel = numHeads * dHead;

    int scatterBytes = (seqLen * dModel * 4) * 2;

    List<GPUTensor<Matrix>>? heads;

    runBenchmarkIsolated(
      name: "SCAT_HEADS",
      iterations: iterations,
      bytesPerOp: scatterBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        heads = <GPUTensor<Matrix>>[];
        for (int i = 0; i < numHeads; i = i + 1) {
          heads!.add(GPUTensor<Matrix>.empty(<int>[seqLen, dHead], allocTape));
        }
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        for (int i = 0; i < heads!.length; i = i + 1) {
          fillTensorGPU(heads![i], 1.0, t);
        }
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        scatterHeadsGPU(heads!, dModel, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int inSize = 4000;
    int pad = 48;
    int outSize = inSize + (2 * pad);

    int padBytes = (inSize * inSize * 4) + (outSize * outSize * 4);

    GPUTensor<Matrix>? matA;

    runBenchmarkIsolated(
      name: "PAD_2D",
      iterations: iterations,
      bytesPerOp: padBytes,
      flopsPerOp: 0,
      allocateInputs: (AllocBuffer allocTape) {
        matA = GPUTensor<Matrix>.empty(<int>[inSize, inSize], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(matA!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        padMatrixGPU(matA!, pad, tape, allocTape);
        return tape;
      },
    );
  }

  // --- NORMALIZATION & REGULARIZATION ---
      {
    int dModel = M;
    int normBytes = (M * M * 4) + (dModel * 4 * 4) + (M * M * 4);
    int normFlops = M * M * 8;

    GPUTensor<Matrix>? input;
    GPUTensor<Vector>? gamma;
    GPUTensor<Vector>? beta;
    GPUTensor<Vector>? mean;
    GPUTensor<Vector>? rstd;

    runBenchmarkIsolated(
      name: "LAYER_NORM",
      iterations: iterations,
      bytesPerOp: normBytes,
      flopsPerOp: normFlops,
      allocateInputs: (AllocBuffer allocTape) {
        input = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
        gamma = GPUTensor<Vector>.empty(<int>[dModel], allocTape);
        beta = GPUTensor<Vector>.empty(<int>[dModel], allocTape);
        mean = GPUTensor<Vector>.empty(<int>[dModel], allocTape);
        rstd = GPUTensor<Vector>.empty(<int>[dModel], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(input!, 1.0, t);
        fillTensorGPU(gamma!, 1.0, t);
        fillTensorGPU(beta!, 0.0, t);
        fillTensorGPU(mean!, 0.0, t);
        fillTensorGPU(rstd!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        layerNormMatrixGPU(
            input!, gamma!, beta!, mean!, rstd!, 1e-5, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int dropBytes = (M * M * 4) + (M * M * 4) + (M * M * 4);
    int dropFlops = M * M;

    GPUTensor<Matrix>? input;

    runBenchmarkIsolated(
      name: "DROPOUT",
      iterations: iterations,
      bytesPerOp: dropBytes,
      flopsPerOp: dropFlops,
      allocateInputs: (AllocBuffer allocTape) {
        input = GPUTensor<Matrix>.empty(<int>[M, M], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(input!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        dropoutGPU<Matrix>(input!, 0.5, tape, allocTape);
        return tape;
      },
    );
  }

  // --- ADVANCED METRICS & LOSSES (Compositional) ---
      {
    int dotBytes = N * 16;
    int dotFlops = N * 2;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "DOT_PROD",
      iterations: iterations,
      bytesPerOp: dotBytes,
      flopsPerOp: dotFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.0, t);
        fillTensorGPU(vecB!, 3.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        dotProductGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int normBytes = N * 12;
    int normFlops = N * 2;

    GPUTensor<Vector>? vecA;

    runBenchmarkIsolated(
      name: "L2_NORM",
      iterations: iterations,
      bytesPerOp: normBytes,
      flopsPerOp: normFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        l2NormGPU(vecA!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int distBytes = N * 24;
    int distFlops = N * 3;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "EUC_DIST",
      iterations: iterations,
      bytesPerOp: distBytes,
      flopsPerOp: distFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 5.0, t);
        fillTensorGPU(vecB!, 2.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        euclideanDistanceGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int cosBytes = N * 40;
    int cosFlops = N * 6;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "COS_SIM",
      iterations: iterations,
      bytesPerOp: cosBytes,
      flopsPerOp: cosFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 1.0, t);
        fillTensorGPU(vecB!, 1.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        cosineSimilarityGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  {
    int maeBytes = N * 24;
    int maeFlops = N * 3;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;

    runBenchmarkIsolated(
      name: "MAE_LOSS",
      iterations: iterations,
      bytesPerOp: maeBytes,
      flopsPerOp: maeFlops,
      allocateInputs: (AllocBuffer allocTape) {
        vecA = GPUTensor<Vector>.empty(<int>[N], allocTape);
        vecB = GPUTensor<Vector>.empty(<int>[N], allocTape);
      },
      initData: (AllocBuffer allocTape) {
        CommandBuffer t = CommandBuffer();
        fillTensorGPU(vecA!, 5.0, t);
        fillTensorGPU(vecB!, 3.0, t);
        GPUEngine.run(t.bytes());
      },
      buildTape: (AllocBuffer allocTape) {
        CommandBuffer tape = CommandBuffer();
        maeLossGPU(vecA!, vecB!, tape, allocTape);
        return tape;
      },
    );
  }

  GPUEngine.dispose();
}

void main() {
  testAll(Target.cuda);
}