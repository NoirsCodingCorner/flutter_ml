import 'dart:typed_data';

import 'package:flutter_ml/gpu_version/ffi/OpCodes.dart';
import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/tensor_math_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';


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
  required void Function() allocateInputs,
  required void Function() initData,
  required CommandBuffer Function() buildTape,
  required void Function() freeAll,
}) {
  Stopwatch allocSw = Stopwatch();

  // 1. Time Allocation of inputs
  allocSw.start();
  allocateInputs();
  allocSw.stop();

  // 2. Initialize data (Excluded from overhead)
  initData();

  // 3. Time Allocation of outputs and tape generation
  allocSw.start();
  CommandBuffer tape = buildTape();
  allocSw.stop();

  Uint8List tapeBytes = tape.bytes();

  // Warmup Engine
  for (int i = 0; i < 3; i = i + 1) {
    CudaEngine.run(tapeBytes);
  }

  // 4. Time Execution
  Stopwatch execSw = Stopwatch();
  execSw.start();
  for (int i = 0; i < iterations; i = i + 1) {
    CudaEngine.run(tapeBytes);
  }
  execSw.stop();

  // 5. Time Deallocation
  Stopwatch freeSw = Stopwatch();
  freeSw.start();
  freeAll();
  freeSw.stop();

  // Calculations
  double seconds = execSw.elapsedMicroseconds / 1000000.0;
  double avgMs = (seconds * 1000.0) / iterations;
  double gbPerSec = (bytesPerOp * iterations / 1000000000.0) / seconds;
  double tflops = (flopsPerOp * iterations / 1000000000000.0) / seconds;

  String timeStr = avgMs.toStringAsFixed(2).padLeft(6);
  String gbStr = gbPerSec.toStringAsFixed(2).padLeft(8);
  String tflopsStr = tflops.toStringAsFixed(4).padLeft(8);
  String allocStr = (allocSw.elapsedMicroseconds / 1000.0).toStringAsFixed(2).padLeft(6);
  String freeStr = (freeSw.elapsedMicroseconds / 1000.0).toStringAsFixed(2).padLeft(6);

  print("[BENCHMARK] " + name.padRight(10) +
      " | Time: " + timeStr + " ms | Bandwidth: " + gbStr + " GB/s | Compute: " + tflopsStr + " TFLOPs" + " Overhead  | Alloc/Tape: " + allocStr + " ms | Free: " + freeStr + " ms");
}

void main() {
  CudaEngine.initialize(debug: false);

  int N = 67108864*2;
  List<int> vecShape = <int>[N];

  int M = 4096 * 2;
  List<int> matShape = <int>[M, M];

  int iterations = 50;

  print("\n==================================================================");
  print("                 CUDA ENGINE PERFORMANCE BENCHMARK                ");
  print("==================================================================");
  print("Vector Size: " + N.toString() + " elements (~" + (N * 4 / 1000000).toStringAsFixed(0) + " MB)");
  print("Matrix Size: " + M.toString() + "x" + M.toString() + " elements");
  print("Iterations:  " + iterations.toString());
  print("Note: VRAM is aggressively wiped and reallocated between each run to test loading speeds of different operations. ");
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
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "ADD",
        iterations: iterations,
        bytesPerOp: binaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
          vecB = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.0, t);
          fillTensorGPU(vecB!, 3.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = addGPU<Vector>(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          out!.free();
        }
    );
  }
  {
    int markovSeqLen = 67108864; // ~67 Million elements
    int numStates = 16;
    int order = 2;

    int buildBytes = markovSeqLen * 4;
    int buildFlops = markovSeqLen;

    GPUTensor<Vector>? sequence;
    GPUTensor<Matrix>? probTable;

    runBenchmarkIsolated(
        name: "MARKOV_TBL",
        iterations: iterations,
        bytesPerOp: buildBytes,
        flopsPerOp: buildFlops,
        allocateInputs: () {
          // 1. Generate random floats in range [-8.0, 8.0] directly in VRAM
          sequence = GPUTensor<Vector>.randomUniform(<int>[markovSeqLen], numStates / 2.0);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();

          // 2. Shift [-8.0, 8.0] to [0.0, 16.0]
          GPUTensor<Vector> shifted = addScalarVectorGPU(sequence!, numStates / 2.0, t);

          // 3. Clamp [0.0, 16.0] to [0.0, 15.0] to guarantee valid state indices
          GPUTensor<Vector> clamped = clampGPU<Vector>(shifted, 0.0, (numStates - 1).toDouble(), t);

          // 4. Overwrite our sequence with the valid randomized states
          t.putInt(OP_COPY);
          t.putString(clamped.id);
          t.putString(sequence!.id);

          CudaEngine.run(t.bytes());

          // Clean up the temporary conversion tensors so VRAM doesn't leak
          shifted.free();
          clamped.free();
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          probTable = buildMarkovTableGPU(sequence!, order, numStates, tape);
          return tape;
        },
        freeAll: () {
          sequence!.free();
          probTable!.free();
        });
  }

  {
    int numStates = 16;
    int order = 2;
    int numHistories = 256; // 16^2

    // 10 Million predictions per pass
    int predictBatchSize = 10000000;

    // Read histories (batch * order * 4), Write Probs (batch * numStates * 4)
    int predictBytes = (predictBatchSize * order * 4) + (predictBatchSize * numStates * 4);
    int predictFlops = predictBatchSize * numStates;

    GPUTensor<Matrix>? historyBatch;
    GPUTensor<Matrix>? probTable;
    GPUTensor<Matrix>? predictions;

    runBenchmarkIsolated(
        name: "MARKOV_PRD",
        iterations: iterations,
        bytesPerOp: predictBytes,
        flopsPerOp: predictFlops,
        allocateInputs: () {
          historyBatch = GPUTensor<Matrix>.empty(<int>[predictBatchSize, order]);
          probTable = GPUTensor<Matrix>.empty(<int>[numHistories, numStates]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(historyBatch!, 1.0, t);
          fillTensorGPU(probTable!, 0.0625, t); // 1/16th uniform prob
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          predictions = markovPredictGPU(historyBatch!, probTable!, numStates, tape);
          return tape;
        },
        freeAll: () {
          historyBatch!.free();
          probTable!.free();
          predictions!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "SUBTRACT",
        iterations: iterations,
        bytesPerOp: binaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
          vecB = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 5.0, t);
          fillTensorGPU(vecB!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = subtractGPU<Vector>(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "MULTIPLY",
        iterations: iterations,
        bytesPerOp: binaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
          vecB = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.0, t);
          fillTensorGPU(vecB!, 3.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = multiplyGPU<Vector>(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "DIVIDE",
        iterations: iterations,
        bytesPerOp: binaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
          vecB = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 10.0, t);
          fillTensorGPU(vecB!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = divideGPU<Vector>(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          out!.free();
        }
    );
  }

  // --- UNARY OPERATIONS ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "ABS",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, -2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = absGPU<Vector>(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "SQRT",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 4.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = sqrtGPU<Vector>(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "LOG",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.718, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = logGPU<Vector>(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "POW",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 3.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = powGPU<Vector>(vecA!, 3.0, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        }
    );
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "CLAMP",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 10.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = clampGPU<Vector>(vecA!, 0.0, 5.0, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        }
    );
  }

  // --- MATMUL ---
      {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? matB;
    GPUTensor<Matrix>? outMatmul;

    runBenchmarkIsolated(
        name: "MATMUL",
        iterations: iterations,
        bytesPerOp: matmulBytes,
        flopsPerOp: matmulFlops,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(matShape);
          matB = GPUTensor<Matrix>.empty(matShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          fillTensorGPU(matB!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outMatmul = GPUTensor<Matrix>.empty(matShape);

          tape.putInt(OP_MATMUL);
          tape.putString(matA!.id);
          tape.putString(matB!.id);
          tape.putString(outMatmul!.id);
          tape.putBool(false);
          tape.putBool(false);
          tape.putFloat(1.0);
          tape.putFloat(0.0);
          tape.putBool(true);

          return tape;
        },
        freeAll: () {
          matA!.free();
          matB!.free();
          outMatmul!.free();
        }
    );
  }
  // --- LINEAR ALGEBRA & BROADCASTING ---

      {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? out;

    runBenchmarkIsolated(
        name: "TRANSPOSE",
        iterations: iterations,
        bytesPerOp: M * M * 8, // Read M (4 bytes), Write Out (4 bytes)
        flopsPerOp: 0, // Zero math, purely memory bound
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = transposeGPU(matA!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "MAT_VEC",
        iterations: iterations,
        bytesPerOp: (M * M * 4) + (M * 4) + (M * 4),
        flopsPerOp: 2 * M * M,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
          vecB = GPUTensor<Vector>.empty(<int>[M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          fillTensorGPU(vecB!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = matVecMulGPU(matA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          vecB!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? vecBias;
    GPUTensor<Matrix>? out;

    runBenchmarkIsolated(
        name: "ADD_BIAS",
        iterations: iterations,
        bytesPerOp: (M * M * 4) + (M * 4) + (M * M * 4),
        flopsPerOp: M * M,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
          vecBias = GPUTensor<Vector>.empty(<int>[M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          fillTensorGPU(vecBias!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = addBiasToMatMulOutGPU(matA!, vecBias!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          vecBias!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? out;

    runBenchmarkIsolated(
        name: "SCALE_MAT",
        iterations: iterations,
        bytesPerOp: M * M * 8, // Read M, Write C
        flopsPerOp: M * M,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = scaleMatrixGPU(matA!, 3.0, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "ADD_SCALAR",
        iterations: iterations,
        bytesPerOp: N * 8, // Using the giant Vector N
        flopsPerOp: N,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]); // Giant Vector size
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = addScalarVectorGPU(vecA!, 5.0, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        });
  }
  // --- ACTIVATIONS ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "RELU",
        iterations: iterations,
        bytesPerOp: unaryBytes, // Read N, Write N
        flopsPerOp: elementFlops, // 1 max() operation per element
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          // Mix of values doesn't matter for memory bandwidth, fill with 1.0
          fillTensorGPU(vecA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = reluGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "SIGMOID",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops * 3, // exp, add, div
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 0.5, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = sigmoidGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "TANH",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops * 3, // hardware dependent, roughly 3 ops
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 0.5, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = vectorTanhGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        });
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? out;

    runBenchmarkIsolated(
        name: "GELU",
        iterations: iterations,
        bytesPerOp: unaryBytes,
        flopsPerOp: elementFlops * 5, // GELU is mathematically heavier
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(vecShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 0.5, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = geluGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          out!.free();
        });
  }

  {
    int softmaxBytes = M * M * 12; // Reads row for max, reads for sum, writes out (rough approx)
    int softmaxFlops = M * M * 3;

    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? out;

    runBenchmarkIsolated(
        name: "SOFTMAX",
        iterations: iterations,
        bytesPerOp: softmaxBytes,
        flopsPerOp: softmaxFlops,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(matShape);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          out = softmaxMatrixGPU(matA!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          out!.free();
        });
  }
  // --- LOSS FUNCTIONS (Memory Bound) ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outLoss;

    runBenchmarkIsolated(
        name: "BCE_LOSS",
        iterations: iterations,
        bytesPerOp: N * 8, // Read Preds (4), Read Targets (4), write 1 scalar
        flopsPerOp: N * 4, // log, mul, add, mul
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 0.5, t); // Preds between 0 and 1
          fillTensorGPU(vecB!, 1.0, t); // Targets 0 or 1
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outLoss = binaryCrossEntropyGPU<Vector>(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outLoss!.free();
        });
  }

  {
    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outLoss;

    runBenchmarkIsolated(
        name: "MSE_VEC",
        iterations: iterations,
        bytesPerOp: N * 8, // Read Preds, Read Targets
        flopsPerOp: N * 3, // sub, square, add (reduction)
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.0, t);
          fillTensorGPU(vecB!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outLoss = mseGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outLoss!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? matB;
    GPUTensor<Scalar>? outLoss;

    runBenchmarkIsolated(
        name: "MSE_MAT",
        iterations: iterations,
        bytesPerOp: M * M * 8,
        flopsPerOp: M * M * 3,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
          matB = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 2.0, t);
          fillTensorGPU(matB!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outLoss = mseMatrixGPU(matA!, matB!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          matB!.free();
          outLoss!.free();
        });
  }

  // --- REDUCTIONS (Memory Bound) ---
      {
    GPUTensor<Vector>? vecA;
    GPUTensor<Scalar>? outSum;

    runBenchmarkIsolated(
        name: "SUM_VEC",
        iterations: iterations,
        bytesPerOp: N * 4, // Read N, write 1
        flopsPerOp: N,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outSum = sumGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          outSum!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? outCols;

    runBenchmarkIsolated(
        name: "SUM_COLS",
        iterations: iterations,
        bytesPerOp: (M * M * 4) + (M * 4), // Read MxM, Write M
        flopsPerOp: M * M,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outCols = sumReduceColumnsGPU(matA!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          outCols!.free();
        });
  }

  {
    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? outRows;

    runBenchmarkIsolated(
        name: "SUM_ROWS",
        iterations: iterations,
        bytesPerOp: (M * M * 4) + (M * 4), // Read MxM, Write M
        flopsPerOp: M * M,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outRows = sumReduceRowsGPU(matA!, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          outRows!.free();
        });
  }

  // --- EMBEDDING LOOKUPS (Memory Scatter/Gather Bound) ---
      {
    int vocabSize = 32000;
    int embedDim = 768; // Standard LLM/BERT hidden size
    int numTokens = 1048576; // 1 Million tokens

    int embedBytes = (numTokens * 4) + // Read indices
        (numTokens * embedDim * 4) + // Gather from weights
        (numTokens * embedDim * 4);  // Write output
    int embedFlops = 0; // Pure memory routing

    GPUTensor<Vector>? indices;
    GPUTensor<Matrix>? weights;
    GPUTensor<Matrix>? outEmbed;

    runBenchmarkIsolated(
        name: "EMBED_VEC",
        iterations: iterations,
        bytesPerOp: embedBytes,
        flopsPerOp: embedFlops,
        allocateInputs: () {
          indices = GPUTensor<Vector>.empty(<int>[numTokens]);
          weights = GPUTensor<Matrix>.empty(<int>[vocabSize, embedDim]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          // We MUST fill indices with a valid vocab index (e.g., 0) so we don't segfault
          fillTensorGPU(indices!, 0.0, t);
          fillTensorGPU(weights!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outEmbed = embeddingLookupGPU(indices!, weights!, tape);
          return tape;
        },
        freeAll: () {
          indices!.free();
          weights!.free();
          outEmbed!.free();
        });
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
    GPUTensor<Tensor3D>? outBatchEmbed;

    runBenchmarkIsolated(
        name: "EMBED_MAT",
        iterations: iterations,
        bytesPerOp: embedBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          batchIndices = GPUTensor<Matrix>.empty(<int>[batchSize, seqLength]);
          weights = GPUTensor<Matrix>.empty(<int>[vocabSize, embedDim]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(batchIndices!, 0.0, t); // Safe valid index
          fillTensorGPU(weights!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outBatchEmbed = embeddingLookupBatchGPU(batchIndices!, weights!, tape);
          return tape;
        },
        freeAll: () {
          batchIndices!.free();
          weights!.free();
          outBatchEmbed!.free();
        });
  }
  // --- TENSOR MANIPULATION & ROUTING (Memory Bound) ---

      {
    int sliceRows = M;
    int sliceCols = M ~/ 2; // Slice half the matrix
    int sliceBytes = (sliceRows * sliceCols * 4) * 2; // Read slice, Write slice

    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? outSlice;

    runBenchmarkIsolated(
        name: "SLICE_COL",
        iterations: iterations,
        bytesPerOp: sliceBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outSlice = sliceColumnGPU(matA!, 0, sliceCols, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          outSlice!.free();
        });
  }

  {
    // Make a very wide matrix so the row extraction moves enough bytes to measure properly
    int wideCols = 1048576; // 1 Million elements per row
    int rowBytes = wideCols * 4 * 2; // Read row, Write vector

    GPUTensor<Matrix>? matA;
    GPUTensor<Vector>? outRow;

    runBenchmarkIsolated(
        name: "SLICE_ROW",
        iterations: iterations,
        bytesPerOp: rowBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[128, wideCols]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outRow = selectRowGPU(matA!, 64, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          outRow!.free();
        });
  }

  {
    int depth = 64;
    int size = 1024;
    int slice3DBytes = (size * size * 4) * 2;

    GPUTensor<Tensor3D>? t3D;
    GPUTensor<Matrix>? outMat;

    runBenchmarkIsolated(
        name: "SLICE_3D",
        iterations: iterations,
        bytesPerOp: slice3DBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          t3D = GPUTensor<Tensor3D>.empty(<int>[depth, size, size]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(t3D!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outMat = selectMatrixFrom3DGPU(t3D!, 32, tape);
          return tape;
        },
        freeAll: () {
          t3D!.free();
          outMat!.free();
        });
  }

  {
    int halfN = N ~/ 2;
    int concatBytes = N * 4 * 2; // Read N total, Write N total

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Vector>? outConcat;

    runBenchmarkIsolated(
        name: "CONCAT_VEC",
        iterations: iterations,
        bytesPerOp: concatBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[halfN]);
          vecB = GPUTensor<Vector>.empty(<int>[halfN]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 1.0, t);
          fillTensorGPU(vecB!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outConcat = concatenateGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outConcat!.free();
        });
  }

  {
    int numMatrices = 16;
    int rows = 1024;
    int cols = 1024;
    int stackBytes = (numMatrices * rows * cols * 4) * 2;

    List<GPUTensor<Matrix>>? matrices;
    GPUTensor<Tensor3D>? outStack;

    runBenchmarkIsolated(
        name: "STACK_MAT",
        iterations: iterations,
        bytesPerOp: stackBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          matrices = List.generate(numMatrices, (_) => GPUTensor<Matrix>.empty(<int>[rows, cols]));
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          for (var m in matrices!) {
            fillTensorGPU(m, 1.0, t);
          }
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outStack = stackMatricesGPU(matrices!, tape);
          return tape;
        },
        freeAll: () {
          for (var m in matrices!) m.free();
          outStack!.free();
        });
  }

  {
    // Simulating Multi-Head Attention Head Scattering
    int seqLen = 4096;
    int dHead = 64;
    int numHeads = 12;
    int dModel = numHeads * dHead;

    int scatterBytes = (seqLen * dModel * 4) * 2; // Gather from 12 heads, write to 1 model tensor

    List<GPUTensor<Matrix>>? heads;
    GPUTensor<Matrix>? outScatter;

    runBenchmarkIsolated(
        name: "SCAT_HEADS",
        iterations: iterations,
        bytesPerOp: scatterBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          heads = List.generate(numHeads, (_) => GPUTensor<Matrix>.empty(<int>[seqLen, dHead]));
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          for (var h in heads!) {
            fillTensorGPU(h, 1.0, t);
          }
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outScatter = scatterHeadsGPU(heads!, dModel, tape);
          return tape;
        },
        freeAll: () {
          for (var h in heads!) h.free();
          outScatter!.free();
        });
  }

  {
    int inSize = 4000;
    int pad = 48; // Total out size = 4096 x 4096
    int outSize = inSize + (2 * pad);

    int padBytes = (inSize * inSize * 4) + (outSize * outSize * 4); // Read inner, write outer (padded zeros are free)

    GPUTensor<Matrix>? matA;
    GPUTensor<Matrix>? outPad;

    runBenchmarkIsolated(
        name: "PAD_2D",
        iterations: iterations,
        bytesPerOp: padBytes,
        flopsPerOp: 0,
        allocateInputs: () {
          matA = GPUTensor<Matrix>.empty(<int>[inSize, inSize]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(matA!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outPad = padMatrixGPU(matA!, pad, tape);
          return tape;
        },
        freeAll: () {
          matA!.free();
          outPad!.free();
        });
  }
  // --- NORMALIZATION & REGULARIZATION ---
      {
    int dModel = M; // 8192
    int normBytes = (M * M * 4) + (dModel * 4 * 4) + (M * M * 4); // Read Input, Read Params (Gamma, Beta, Mean, Var), Write Output
    int normFlops = M * M * 8; // Mean, Var, Normalize, Scale, Shift

    GPUTensor<Matrix>? input;
    GPUTensor<Vector>? gamma;
    GPUTensor<Vector>? beta;
    GPUTensor<Vector>? mean;
    GPUTensor<Vector>? rstd;
    GPUTensor<Matrix>? outNorm;

    runBenchmarkIsolated(
        name: "LAYER_NORM",
        iterations: iterations,
        bytesPerOp: normBytes,
        flopsPerOp: normFlops,
        allocateInputs: () {
          input = GPUTensor<Matrix>.empty(<int>[M, M]);
          gamma = GPUTensor<Vector>.empty(<int>[dModel]);
          beta = GPUTensor<Vector>.empty(<int>[dModel]);
          mean = GPUTensor<Vector>.empty(<int>[dModel]);
          rstd = GPUTensor<Vector>.empty(<int>[dModel]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(input!, 1.0, t);
          fillTensorGPU(gamma!, 1.0, t);
          fillTensorGPU(beta!, 0.0, t);
          fillTensorGPU(mean!, 0.0, t);
          fillTensorGPU(rstd!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outNorm = layerNormMatrixGPU(input!, gamma!, beta!, mean!, rstd!, 1e-5, tape);
          return tape;
        },
        freeAll: () {
          input!.free();
          gamma!.free();
          beta!.free();
          mean!.free();
          rstd!.free();
          outNorm!.free();
        });
  }

  {
    int dropBytes = (M * M * 4) + (M * M * 4) + (M * M * 4); // Read Input, Write Mask, Write Output
    int dropFlops = M * M; // Random RNG comparison per element

    GPUTensor<Matrix>? input;
    GPUTensor<Matrix>? outDrop;

    runBenchmarkIsolated(
        name: "DROPOUT",
        iterations: iterations,
        bytesPerOp: dropBytes,
        flopsPerOp: dropFlops,
        allocateInputs: () {
          input = GPUTensor<Matrix>.empty(<int>[M, M]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(input!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outDrop = dropoutGPU<Matrix>(input!, 0.5, tape);
          return tape;
        },
        freeAll: () {
          input!.free();
          outDrop!.free();
        });
  }
  // --- ADVANCED METRICS & LOSSES (Compositional) ---
      {
    int dotBytes = N * 16; // Multiply (Read 2N, Write 1N), Sum (Read 1N, Write 1)
    int dotFlops = N * 2;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outDot;

    runBenchmarkIsolated(
        name: "DOT_PROD",
        iterations: iterations,
        bytesPerOp: dotBytes,
        flopsPerOp: dotFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.0, t);
          fillTensorGPU(vecB!, 3.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outDot = dotProductGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outDot!.free();
        });
  }

  {
    int normBytes = N * 12; // Pow (Read 1N, Write 1N), Sum (Read 1N, Write 1)
    int normFlops = N * 2;

    GPUTensor<Vector>? vecA;
    GPUTensor<Scalar>? outNorm;

    runBenchmarkIsolated(
        name: "L2_NORM",
        iterations: iterations,
        bytesPerOp: normBytes,
        flopsPerOp: normFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outNorm = l2NormGPU(vecA!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          outNorm!.free();
        });
  }

  {
    int distBytes = N * 24; // Sub (3N), Pow (2N), Sum (1N)
    int distFlops = N * 3;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outDist;

    runBenchmarkIsolated(
        name: "EUC_DIST",
        iterations: iterations,
        bytesPerOp: distBytes,
        flopsPerOp: distFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 5.0, t);
          fillTensorGPU(vecB!, 2.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outDist = euclideanDistanceGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outDist!.free();
        });
  }

  {
    int cosBytes = N * 40; // Dot (4N), NormA (3N), NormB (3N) -> Total 10N reads/writes
    int cosFlops = N * 6;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outCos;

    runBenchmarkIsolated(
        name: "COS_SIM",
        iterations: iterations,
        bytesPerOp: cosBytes,
        flopsPerOp: cosFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 1.0, t);
          fillTensorGPU(vecB!, 1.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outCos = cosineSimilarityGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outCos!.free();
        });
  }

  {
    int maeBytes = N * 24; // Sub (3N), Abs (2N), Sum (1N)
    int maeFlops = N * 3;

    GPUTensor<Vector>? vecA;
    GPUTensor<Vector>? vecB;
    GPUTensor<Scalar>? outMae;

    runBenchmarkIsolated(
        name: "MAE_LOSS",
        iterations: iterations,
        bytesPerOp: maeBytes,
        flopsPerOp: maeFlops,
        allocateInputs: () {
          vecA = GPUTensor<Vector>.empty(<int>[N]);
          vecB = GPUTensor<Vector>.empty(<int>[N]);
        },
        initData: () {
          CommandBuffer t = CommandBuffer();
          fillTensorGPU(vecA!, 5.0, t);
          fillTensorGPU(vecB!, 3.0, t);
          CudaEngine.run(t.bytes());
        },
        buildTape: () {
          CommandBuffer tape = CommandBuffer();
          outMae = maeLossGPU(vecA!, vecB!, tape);
          return tape;
        },
        freeAll: () {
          vecA!.free();
          vecB!.free();
          outMae!.free();
        });
  }



  CudaEngine.dispose();
}
