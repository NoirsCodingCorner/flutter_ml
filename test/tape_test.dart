
import 'package:flutter_ml/full_library.dart';

Future<void> main() async {
  // =========================================================================
  // 0. ENGINE INITIALIZATION
  // =========================================================================
  await GPUEngine.initialize(debug: false, target: Target.auto);

  // =========================================================================
  // 1. CREATE THE EXPLICIT TAPES
  // =========================================================================
  // Memory lifecycle tape (owns every VRAM allocation & deallocation)
  AllocBuffer allocTape = AllocBuffer();

  // Execution tapes (pure byte streams executed by GPUEngine)
  CommandBuffer zeroGradTape = CommandBuffer();
  CommandBuffer forwardTape = CommandBuffer();
  CommandBuffer backwardTape = CommandBuffer();
  CommandBuffer optimizeTape = CommandBuffer();

  // =========================================================================
  // 2. PHASE 1: GRAPH CONSTRUCTION (Runs Once)
  // =========================================================================
  // Standard XOR dataset: 4 samples x 2 features -> 4 samples x 1 target
  Matrix xorInputs = <List<double>>[
    <double>[0.0, 0.0],
    <double>[0.0, 1.0],
    <double>[1.0, 0.0],
    <double>[1.0, 1.0],
  ];

  Matrix xorTargets = <List<double>>[
    <double>[0.0],
    <double>[1.0],
    <double>[1.0],
    <double>[0.0],
  ];

  // Input needs grad because DenseReluTL's backward pass computes dX.
  // Target does not need grad because MSE backward only writes dPred.
  GPUTensor<Matrix> input = GPUTensor<Matrix>(
    xorInputs,
    allocTape,
    id: 'xor_input',
    sourceOp: 'dataset:input',
  );
  GPUTensor<Matrix> target = GPUTensor<Matrix>(
    xorTargets,
    allocTape,
    id: 'xor_target',
    requiresGrad: false,
    sourceOp: 'dataset:target',
  );

  // Define a simple 2-layer Dense network: [4, 2] -> [4, 8] -> [4, 1] -> Sigmoid
  List<TapeLayer> layers = <TapeLayer>[
    DenseReluTL(8),
    DenseTL(1),
    SigmoidMatrixTL(),
  ];

  // Record forward pass into forwardTape and all layer weights/intermediates into allocTape
  GPUTensor<dynamic> current = input;
  for (int i = 0; i < layers.length; i = i + 1) {
    current = layers[i].call(current, forwardTape, allocTape);
  }
  GPUTensor<Matrix> predictions = current as GPUTensor<Matrix>;

  // Record MSE loss at the end of forwardTape
  GPUTensor<Scalar> loss =
  mseMatrixGPU(predictions, target, forwardTape, allocTape);

  // Compile backward pass once from the loss graph into backwardTape
  loss.backward(backwardTape);

  // Collect trainable parameters and compile optimizer into optimizeTape
  List<GPUTensor> trainableParams = <GPUTensor>[];
  for (int i = 0; i < layers.length; i = i + 1) {
    List<GPUTensor> layerParams = layers[i].parameters;
    for (int j = 0; j < layerParams.length; j = j + 1) {
      trainableParams.add(layerParams[j]);
    }
  }

  double learningRate = 0.05;
  AdamGPU optimizer = AdamGPU(trainableParams, learningRate, allocTape);
  optimizer.step(optimizeTape);

  // Compile zeroGradTape directly from every grad allocation recorded in allocTape
  allocTape.compileZeroGradTape(zeroGradTape);

  // =========================================================================
  // 3. INSPECT ALLOCATIONS & COMPILED TAPES (Before Touching VRAM)
  // =========================================================================
  allocTape.logBuffer(verbose: true);

  Logger.cyan('=== COMPILED EXECUTION TAPES ===', prefix: '');
  Logger.blue(
    'zeroGradTape : ${zeroGradTape.offset} bytes | patches: ${zeroGradTape.patchOffsets.keys.toList()}',
    prefix: '',
  );
  Logger.blue(
    'forwardTape  : ${forwardTape.offset} bytes | patches: ${forwardTape.patchOffsets.keys.toList()}',
    prefix: '',
  );
  Logger.blue(
    'backwardTape : ${backwardTape.offset} bytes | patches: ${backwardTape.patchOffsets.keys.toList()}',
    prefix: '',
  );
  Logger.blue(
    'optimizeTape : ${optimizeTape.offset} bytes | patches: ${optimizeTape.patchOffsets.keys.toList()}',
    prefix: '',
  );

  // =========================================================================
  // 4. PHASE 2: ALLOCATE VRAM & RUN TRAINING LOOP (Tapes Only)
  // =========================================================================
  // Nothing existed in VRAM until this single call:
  allocTape.execute();

  List<String> allocatedNames = GPUEngine.getTensorNames();
  Logger.green(
    'VRAM live tensors after allocTape.execute(): ${allocatedNames.length}',
    prefix: '',
  );

  // From here on, neither layers nor AdamGPU are called—only the tapes and I/O handles.
  int epochs = 500;
  for (int step = 1; step <= epochs; step = step + 1) {
    // Patch dynamic per-step values directly in the compiled tape
    optimizeTape.patchInt('step', step);
    optimizeTape.patchFloat('lr', learningRate);

    // Execute the 4 compiled tapes in order
    GPUEngine.run(zeroGradTape.bytes());
    GPUEngine.run(forwardTape.bytes());
    GPUEngine.run(backwardTape.bytes());
    GPUEngine.run(optimizeTape.bytes());

    // Pull scalar loss data only when logging
    if (step == 1 || step % 50 == 0) {
      loss.dataToCpu();
      Logger.yellow(
        'Step ${step.toString().padLeft(4)} | Loss: ${loss.value.toStringAsFixed(6)}',
        prefix: '',
      );
    }
  }

  // =========================================================================
  // 5. INFERENCE CHECK (Replay forwardTape Only + Object Permanence)
  // =========================================================================
  // Push fresh input data in place into the existing input handle (no reallocation)
  List<double> flatTestInput = <double>[
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
  ];
  input.pushData(flatTestInput);

  // Run only the forward tape and inspect predictions via the retained handle
  GPUEngine.run(forwardTape.bytes());
  predictions.dataToCpu();

  Matrix finalPreds = predictions.value;
  Logger.cyan('=== FINAL PREDICTIONS ===', prefix: '');
  for (int i = 0; i < xorInputs.length; i = i + 1) {
    Logger.green(
      'Input: ${xorInputs[i]} -> Pred: ${finalPreds[i][0].toStringAsFixed(4)} (Target: ${xorTargets[i][0]})',
      prefix: '',
    );
  }

  // =========================================================================
  // 6. TEARDOWN (allocTape Owns Cleanup)
  // =========================================================================
  allocTape.freeAll();

  List<String> remainingNames = GPUEngine.getTensorNames();
  Logger.cyan(
    'VRAM live tensors after allocTape.freeAll(): ${remainingNames.length}',
    prefix: '',
  );

  GPUEngine.dispose();
}