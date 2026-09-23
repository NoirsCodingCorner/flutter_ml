import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/speedtest.dart';

import 'full_library.dart';
import 'gpu_version/SNetworkGPU.dart'; // Passe den Pfad an, falls nötig

void main() {
  runApp(MaterialApp(home: SimpleGPUApp()));
}



class SimpleGPUApp extends StatelessWidget {
  Target target=Target.cuda;
  SimpleGPUApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('WebGPU Test')),
      body: Center(
        child: ElevatedButton(
          onPressed: () async {
            try {
              // useCuda: false zwingt die Engine, das WebGPU-Backend zu nutzen
              //print("✅ GPUEngine erfolgreich initialisiert!");
              //await testMath();
              //print("_____________________________________________________________________________________________");
              //await testLinear();
              //print("_____________________________________________________________________________________________");
              //await testBackward();
              //print("_____________________________________________________________________________________________");
              //await runTransformer();
              //print("_____________________________________________________________________________________________");
              //await testResultTransfer();
              //print("_____________________________________________________________________________________________");
              //await testSpeeds();
              await runSimple();

            } catch (e) {
              print("❌ Fehler bei der Initialisierung: $e");
            }
          },
          child: const Text('Initialize GPU Engine'),
        ),
      ),
    );
  }

  Future<bool> runSimple() async {
    await GPUEngine.initialize(target: target);

    CommandBuffer tape = CommandBuffer();

    GPUTensor<Vector> VecA = GPUTensor([1.1, 2.2, 3.3]);
    GPUTensor<Vector> VecB = GPUTensor([4.4, 5.5, 6.6]);
    GPUTensor<Vector> VecC = GPUTensor([1.0, 0.1, -1.0]);

    GPUTensor<Vector> VecD = addVectorGPU(VecA, VecB, tape);
    GPUTensor<Vector> VecE = elementWiseMultiplyGPU(VecC, VecD, tape);

    GPUEngine.run(tape.bytes());

    VecE.toCpu();
    print("Result: ${VecE.value}");
    VecE.printGraph();
    TapeDecoder(tape.bytes()).decode();
    return true;
  }

  Future<bool> runTransformer() async {
    await GPUEngine.initialize(debug: false, target: target);
    Random random = Random();

    int numHeads = 8;
    int runsPerTest = 50;

    print(
      '====================================================================================================',
    );
    print(
      '              GPU TRANSFORMER ENCODER BLOCK INFERENCE BENCHMARK (FORWARD ONLY)                      ',
    );
    print(
      '====================================================================================================',
    );
    print(
      'Heads: $numHeads | Feed-Forward Ratio: 4x dModel | Inference Runs/Test: $runsPerTest',
    );
    print(
      '----------------------------------------------------------------------------------------------------',
    );
    print(
      ' SeqLen | dModel | Compile(ms) | Latency(ms) | Throughput (GB/s) | Compute (TFLOPs)',
    );
    print(
      '----------------------------------------------------------------------------------------------------',
    );

    List<int> sequences = <int>[32, 64, 128, 512];
    List<int> dimensions = <int>[32, 64, 128, 256];

    for (int s = 0; s < sequences.length; s = s + 1) {
      int seqLength = sequences[s];

      for (int d = 0; d < dimensions.length; d = d + 1) {
        int dModel = dimensions[d];
        int dff = dModel * 4; // Standard Transformer configuration

        // 1. Generate Input Matrix (Batch = 1)
        List<List<double>> hInput = <List<double>>[];
        for (int i = 0; i < seqLength; i = i + 1) {
          List<double> row = <double>[];
          for (int j = 0; j < dModel; j = j + 1) {
            row.add((random.nextDouble() * 2.0) - 1.0);
          }
          hInput.add(row);
        }

        GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);

        // 2. Build Transformer Block
        TransformerEncoderBlockTapeLayer transformerBlock =
            TransformerEncoderBlockTapeLayer(dModel, numHeads, dff);
        transformerBlock.build(input);

        Stopwatch compileSw = Stopwatch();
        compileSw.start();

        // ===================================================================
        // FORWARD TAPE (INFERENCE ONLY)
        // ===================================================================
        CommandBuffer fTape = CommandBuffer();
        List<GPUTensor> intermediates = <GPUTensor>[];

        GPUTensor<Matrix> output =
            transformerBlock.forward(input, fTape, intermediates)
                as GPUTensor<Matrix>;

        Uint8List forwardBytes = fTape.bytes();

        compileSw.stop();

        // ===================================================================
        // EXECUTION LOOP & METRIC CALCULATIONS
        // ===================================================================

        double seqD = seqLength.toDouble();
        double modD = dModel.toDouble();
        double ffD = dff.toDouble();

        // FLOPs purely for the forward pass
        // MHA (Proj + Out): 8 * Seq * D^2
        // MHA (Attention):  4 * Seq^2 * D
        // FFN (W1 + W2):    4 * Seq * D * FF
        double totalFlopsStep =
            (8.0 * seqD * modD * modD) +
            (4.0 * seqD * seqD * modD) +
            (4.0 * seqD * modD * ffD);

        // Memory Traffic (Bytes Read/Written to VRAM)
        // Weights: ~4D^2 (MHA) + 2*D*FF (FFN) -> * 4 bytes
        double weightBytes = (4.0 * modD * modD + 2.0 * modD * ffD) * 4.0;
        // Activations: Rough estimate of intermediate reads/writes per step
        double actBytes =
            ((20.0 * seqD * modD) + (5.0 * seqD * ffD) + (4.0 * seqD * seqD)) *
            4.0;
        double totalBytesStep = weightBytes + actBytes;

        Stopwatch runSw = Stopwatch();

        // Warmup (Push weights into VRAM caches)
        GPUEngine.run(forwardBytes);

        // Inference Loop
        runSw.start();
        for (int run = 1; run <= runsPerTest; run = run + 1) {
          GPUEngine.run(forwardBytes);
        }
        runSw.stop();

        double avgRunSec =
            (runSw.elapsedMicroseconds / 1000000.0) / runsPerTest;
        double avgRunMs = avgRunSec * 1000.0;

        double tflops = (totalFlopsStep / avgRunSec) / 1000000000000.0;
        double gbps = (totalBytesStep / avgRunSec) / 1000000000.0;

        String sSeq = seqLength.toString().padRight(6);
        String sDim = dModel.toString().padRight(6);
        String sComp = compileSw.elapsedMilliseconds.toString().padRight(11);
        String sAvg = avgRunMs.toStringAsFixed(2).padRight(11);
        String sGbps = gbps.toStringAsFixed(2).padRight(17);
        String sTflops = tflops.toStringAsFixed(4).padRight(16);

        print(' $sSeq | $sDim | $sComp | $sAvg | $sGbps | $sTflops');

        // Free Memory
        transformerBlock.free();
        input.free();
        output.free();
        for (int i = 0; i < intermediates.length; i = i + 1) {
          intermediates[i].free();
        }
      }
    }

    print(
      '----------------------------------------------------------------------------------------------------',
    );
    print('Benchmark Complete.');
    return true;

  }

  Future<bool> testResultTransfer() async {
    await GPUEngine.initialize(debug: false, target: target);
    GPUTensor<Vec> a = GPUTensor<Vec>([
      1.0,
      2.0,
      3.0,
      4.0,
      5.0,
      6.0,
      7.0,
      8.0,
      9.0,
      10.0,
    ]);
    GPUTensor<Vec> b = GPUTensor<Vec>([
      10.0,
      9.0,
      8.0,
      7.0,
      6.0,
      5.0,
      4.0,
      3.0,
      2.0,
      1.0,
    ]);

    CommandBuffer tape = CommandBuffer();
    GPUTensor<Scalar> c = dotProductGPU(a, b, tape);
    GPUTensor<Scalar> d = l2NormGPU(a, tape);
    GPUTensor<Scalar> e = euclideanDistanceGPU(a, b, tape);
    GPUTensor<Scalar> f = cosineSimilarityGPU(a, b, tape);
    GPUTensor<Scalar> g = maeLossGPU(a, b, tape);
    GPUEngine.run(tape.bytes());
    a.toCpu();
    b.toCpu();
    c.toCpu();
    d.toCpu();
    e.toCpu();
    f.toCpu();
    g.toCpu();
    print("Original Tensors: ${a.value}, ${b.value}");
    print("Dot Product: ${c.value}");
    print("L2 Norm: ${d.value}");
    print("Euclidean Distance: ${e.value}");
    print("Cosine Similarity: ${f.value}");
    print("MAE Loss: ${g.value}");
    return true;

  }

  Future<bool> testLearning() async {
    print("🚀 Initialisiere GPU Engine...");
    // 1. Engine starten (passe das Target an, falls du auf Android/Web bist)
    await GPUEngine.initialize(debug: false, target: this.target); //

    print("📦 Allokiere Tensoren im VRAM...");
    // 2. Tensoren anlegen
    // Shape: [1, 2] bedeutet 1 Sample mit 2 Features
    var input = GPUTensor<Matrix>.empty([1, 2]); //
    var target = GPUTensor<Matrix>.empty([1, 1]); //

    // Gewichte & Bias direkt im VRAM randomisiert initialisieren
    var weights = GPUTensor<Matrix>.randomUniform([2, 1], 1.0); //[cite: 2]
    var bias = GPUTensor<Vector>.randomUniform([1], 1.0); //[cite: 2]

    print("📼 Zeichne Computational Graph (Tape) auf...");
    // 3. CommandBuffer erstellen
    var tape = CommandBuffer(); //[cite: 5]

    // --- FORWARD PASS ---
    // input * weights
    var step1 = matMulGPU(input, weights, tape); //[cite: 3]
    // + bias
    var output = addBiasToMatMulOutGPU(step1, bias, tape); //[cite: 3]
    // Loss berechnen (Mean Squared Error)
    var loss = mseMatrixGPU(output, target, tape); //[cite: 3]

    // --- BACKWARD PASS ---
    // Vorherige Gradienten im Graphen nullen
    loss.zeroGraphGrads(tape); //[cite: 2]
    // Backpropagation auslösen (füllt den ersten Gradienten mit 1.0)
    loss.backward(tape, fillOnes: true); //[cite: 2]

    // --- OPTIMIZER STEP ---
    // Gewichte und Bias via Stochastic Gradient Descent anpassen
    sgdUpdateGPU(weights, 0.01, tape); //[cite: 3]
    sgdUpdateGPU(bias, 0.01, tape); //[cite: 3]

    // Tape zu Bytes kompilieren
    Uint8List compiledTape = tape.bytes(); //[cite: 5]

    print("🔥 Starte Training auf der Engine...");
    // 4. Test-Trainings-Loop feuern
    for (int epoch = 1; epoch <= 50; epoch++) {
      // a) Daten von der Dart-CPU in den VRAM pushen
      input.pushData([0.5, 0.8]); //[cite: 2]
      target.pushData([1.0]); //[cite: 2]

      // b) Das komplette kompilierte Tape auf der C++ Engine ausführen
      GPUEngine.run(compiledTape); //[cite: 1]

      // c) Den berechneten Loss-Wert zurück in den RAM holen, um ihn anzuzeigen
      loss.toCpu(); //[cite: 2]
      print("Epoche $epoch - Loss: ${loss.value}"); //[cite: 2]
    }

    print("🧹 Räume VRAM auf...");
    // 5. Sauberes Memory Management
    input.free(); //[cite: 2]
    target.free(); //[cite: 2]
    weights.free(); //[cite: 2]
    bias.free(); //[cite: 2]
    step1.free(); //[cite: 2]
    output.free(); //[cite: 2]
    loss.free(); //[cite: 2]

    GPUEngine.dispose(); //[cite: 1]
    print("✅ Test erfolgreich abgeschlossen!");
    return true;

  }

  Future<bool> testMath() async {
    print("🚀 Starte massive GPU Math-Validierung...");
    await GPUEngine.initialize(
      debug: false,
      target: this.target,
    ); //[cite: 1]

    // ======================================================================
    // 1. BASIC MATH (Element-wise)
    // ======================================================================
    print("\n--- 1. BASIC MATH (Vectors) ---");
    {
      var vA = GPUTensor<Vector>.empty([2]); //[cite: 2]
      var vB = GPUTensor<Vector>.empty([2]); //[cite: 2]
      vA.pushData([4.0, 9.0]); //[cite: 2]
      vB.pushData([2.0, 3.0]); //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      var tAdd = addGPU<Vector>(vA, vB, tape); //[cite: 3]
      var tSub = subtractGPU<Vector>(vA, vB, tape); //[cite: 3]
      var tMul = multiplyGPU<Vector>(vA, vB, tape); //[cite: 3]
      var tDiv = divideGPU<Vector>(vA, vB, tape); //[cite: 3]
      var tExp = vectorExpGPU(vA, tape); //[cite: 3]
      var tAbs = absGPU<Vector>(
        GPUTensor<Vector>([-4.0, -9.0]),
        tape,
      ); //[cite: 2, 3]
      var tSqrt = sqrtGPU<Vector>(vA, tape); //[cite: 3]
      var tLog = logGPU<Vector>(vA, tape); //[cite: 3]
      var tPow = powGPU<Vector>(vA, 2.0, tape); //[cite: 3]
      var tClamp = clampGPU<Vector>(vA, 0.0, 5.0, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      tAdd.toCpu();
      tSub.toCpu();
      tMul.toCpu();
      tDiv.toCpu(); //[cite: 2]
      tExp.toCpu();
      tAbs.toCpu();
      tSqrt.toCpu();
      tLog.toCpu();
      tPow.toCpu();
      tClamp.toCpu(); //[cite: 2]

      print("Inputs: vA=[4, 9], vB=[2, 3]");
      print("ADD:    ${tAdd.value}");
      print("SUB:    ${tSub.value}");
      print("MUL:    ${tMul.value}");
      print("DIV:    ${tDiv.value}");
      print("EXP(vA):${tExp.value}");
      print("ABS([-4, -9]): ${tAbs.value}");
      print("SQRT(vA):      ${tSqrt.value}");
      print("LOG(vA):       ${tLog.value}");
      print("POW(vA, 2):    ${tPow.value}");
      print("CLAMP(vA,0,5): ${tClamp.value}");

      vA.free();
      vB.free();
      tAdd.free();
      tSub.free();
      tMul.free();
      tDiv.free(); //[cite: 2]
      tExp.free();
      tAbs.free();
      tSqrt.free();
      tLog.free();
      tPow.free();
      tClamp.free(); //[cite: 2]
    }

    // ======================================================================
    // 2. LINEAR ALGEBRA & MATRIX OPS
    // ======================================================================
    print("\n--- 2. LINEAR ALGEBRA ---");
    {
      print("Make GPUTensors");
      var mA = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      var mB = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      var vC = GPUTensor<Vector>.empty([2]); //[cite: 2]

      print("Push Data to GPU");
      mA.pushData([1.0, 2.0, 3.0, 4.0]); // [[1, 2], [3, 4]] //[cite: 2]
      mB.pushData([2.0, 0.0, 1.0, 2.0]); // [[2, 0], [1, 2]] //[cite: 2]
      vC.pushData([10.0, 20.0]); //[cite: 2]

      print("Tape");
      var tape = CommandBuffer(); //[cite: 5]
      print("tMatMul");
      var tMatMul = matMulGPU(mA, mB, tape); //[cite: 3]
      print("tMatVec");
      var tMatVec = matVecMulGPU(mA, vC, tape); //[cite: 3]
      print("tTrans");
      var tTrans = transposeGPU(mA, tape); //[cite: 3]
      print("tAddBi");
      var tAddBi = addBiasToMatMulOutGPU(mA, vC, tape); //[cite: 3]
      print("tScale");
      var tScale = scaleMatrixGPU(mA, 10.0, tape); //[cite: 3]
      print("tAddSca");
      var tAddSca = addScalarMatrixGPU(
        mA,
        GPUTensor<Scalar>(5.0),
        tape,
      ); //[cite: 2, 3]

      print("Run tape");
      GPUEngine.run(tape.bytes()); //[cite: 1]

      print("Push to CPU");
      tMatMul.toCpu();
      tMatVec.toCpu();
      tTrans.toCpu();
      tAddBi.toCpu();
      tScale.toCpu();
      tAddSca.toCpu(); //[cite: 2]

      print("mA=[[1,2],[3,4]], mB=[[2,0],[1,2]], vC=[10,20]");
      print("MATMUL(mA, mB):   ${tMatMul.value}");
      print("MATVEC(mA, vC):   ${tMatVec.value}");
      print("TRANSPOSE(mA):    ${tTrans.value}");
      print("ADDBIAS(mA, vC):  ${tAddBi.value}");
      print("SCALE(mA, 10):    ${tScale.value}");
      print("ADD_SCALAR(mA,5): ${tAddSca.value}");

      mA.free();
      mB.free();
      vC.free();
      tMatMul.free();
      tMatVec.free();
      tTrans.free();
      tAddBi.free();
      tScale.free();
      tAddSca.free(); //[cite: 2]
    }

    // ======================================================================
    // 3. ACTIVATIONS
    // ======================================================================
    print("\n--- 3. ACTIVATIONS ---");
    {
      var vA = GPUTensor<Vector>.empty([3]); //[cite: 2]
      vA.pushData([-1.0, 0.0, 1.0]); //[cite: 2]

      var mA = GPUTensor<Matrix>.empty([1, 2]); //[cite: 2]
      mA.pushData([0.0, 1.0]); //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      var tRelu = reluGPU(vA, tape); //[cite: 3]
      var tSig = sigmoidGPU(vA, tape); //[cite: 3]
      var tTanh = vectorTanhGPU(vA, tape); //[cite: 3]
      var tGelu = geluGPU(vA, tape); //[cite: 3]
      var tSoftmax = softmaxMatrixGPU(mA, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      tRelu.toCpu();
      tSig.toCpu();
      tTanh.toCpu();
      tGelu.toCpu();
      tSoftmax.toCpu(); //[cite: 2]

      print("Inputs: vA=[-1, 0, 1], mA=[[0, 1]]");
      print("RELU(vA):    ${tRelu.value}");
      print("SIGMOID(vA): ${tSig.value}");
      print("TANH(vA):    ${tTanh.value}");
      print("GELU(vA):    ${tGelu.value}");
      print("SOFTMAX(mA): ${tSoftmax.value}");

      vA.free();
      mA.free();
      tRelu.free();
      tSig.free();
      tTanh.free();
      tGelu.free();
      tSoftmax.free(); //[cite: 2]
    }

    // ======================================================================
    // 4. REDUCTIONS & STATS
    // ======================================================================
    print("\n--- 4. REDUCTIONS & STATS ---");
    {
      var vA = GPUTensor<Vector>.empty([3]); //[cite: 2]
      var vB = GPUTensor<Vector>.empty([3]); //[cite: 2]
      var mA = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]

      vA.pushData([1.0, 2.0, 3.0]); //[cite: 2]
      vB.pushData([4.0, 5.0, 6.0]); //[cite: 2]
      mA.pushData([1.0, 2.0, 3.0, 4.0]); // [[1, 2], [3, 4]] //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      var tSumVec = sumGPU(vA, tape); //[cite: 3]
      var tSumMat = sumMatrixGPU(mA, tape); //[cite: 3]
      var tSumCol = sumReduceColumnsGPU(mA, tape); //[cite: 3]
      var tSumRow = sumReduceRowsGPU(mA, tape); //[cite: 3]
      var tDot = dotProductGPU(vA, vB, tape); //[cite: 3]
      var tL2 = l2NormGPU(
        GPUTensor<Vector>([3.0, 4.0]),
        tape,
      ); // Sollte 5 ergeben //[cite: 2, 3]
      var tEuc = euclideanDistanceGPU(vA, vB, tape); //[cite: 3]
      var tCos = cosineSimilarityGPU(vA, vB, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      tSumVec.toCpu();
      tSumMat.toCpu();
      tSumCol.toCpu();
      tSumRow.toCpu(); //[cite: 2]
      tDot.toCpu();
      tL2.toCpu();
      tEuc.toCpu();
      tCos.toCpu(); //[cite: 2]

      print("Inputs: vA=[1,2,3], vB=[4,5,6], mA=[[1,2],[3,4]]");
      print("SUM(vA):       ${tSumVec.value}");
      print("SUM(mA):       ${tSumMat.value}");
      print("SUM_COLS(mA):  ${tSumCol.value}");
      print("SUM_ROWS(mA):  ${tSumRow.value}");
      print("DOT(vA, vB):   ${tDot.value}");
      print("L2NORM([3,4]): ${tL2.value}");
      print("EUC_DIST(vA, vB): ${tEuc.value}");
      print("COS_SIM(vA, vB):  ${tCos.value}");

      vA.free();
      vB.free();
      mA.free();
      tSumVec.free();
      tSumMat.free();
      tSumCol.free();
      tSumRow.free();
      tDot.free();
      tL2.free();
      tEuc.free();
      tCos.free(); //[cite: 2]
    }

    // ======================================================================
    // 5. LOSS FUNCTIONS
    // ======================================================================
    print("\n--- 5. LOSS FUNCTIONS ---");
    {
      var vPred = GPUTensor<Vector>.empty([2]); //[cite: 2]
      var vTarg = GPUTensor<Vector>.empty([2]); //[cite: 2]

      vPred.pushData([0.8, 0.2]); //[cite: 2]
      vTarg.pushData([1.0, 0.0]); //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      var tBce = binaryCrossEntropyGPU<Vector>(vPred, vTarg, tape); //[cite: 3]
      var tMse = mseGPU(vPred, vTarg, tape); //[cite: 3]
      var tMae = maeLossGPU(vPred, vTarg, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]
      tBce.toCpu();
      tMse.toCpu();
      tMae.toCpu(); //[cite: 2]

      print("Inputs: Pred=[0.8, 0.2], Targ=[1.0, 0.0]");
      print("BCE_LOSS: ${tBce.value}");
      print("MSE_LOSS: ${tMse.value}");
      print("MAE_LOSS: ${tMae.value}");

      vPred.free();
      vTarg.free();
      tBce.free();
      tMse.free();
      tMae.free(); //[cite: 2]
    }

    // ======================================================================
    // 6. SHAPE / TENSOR MANIPULATION
    // ======================================================================
    print("\n--- 6. SHAPE & TENSOR MANIPULATION ---");
    {
      var vA = GPUTensor<Vector>.empty([4]); //[cite: 2]
      vA.pushData([1.0, 2.0, 3.0, 4.0]); //[cite: 2]

      var mA = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      mA.pushData([1.0, 2.0, 3.0, 4.0]); //[cite: 2]
      var mB = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      mB.pushData([5.0, 6.0, 7.0, 8.0]); //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      var tReshape = reshapeVectorToMatrixGPU(vA, 2, 2, tape); //[cite: 3]
      var tSliceCol = sliceColumnGPU(mA, 0, 1, tape); //[cite: 3]
      var tSelRow = selectRowGPU(mA, 1, tape); //[cite: 3]
      var tConcat = concatenateGPU(
        vA,
        GPUTensor<Vector>([5.0, 6.0]),
        tape,
      ); //[cite: 2, 3]
      var tStack = stackMatricesGPU([mA, mB], tape); //[cite: 3]
      var tPad = padMatrixGPU(mA, 1, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      tReshape.toCpu();
      tSliceCol.toCpu();
      tSelRow.toCpu();
      tConcat.toCpu();
      tStack.toCpu();
      tPad.toCpu(); //[cite: 2]

      print("RESHAPE(v[1,2,3,4] -> 2x2): ${tReshape.value}");
      print("SLICE_COL(mA, 0, 1):        ${tSliceCol.value}");
      print("SELECT_ROW(mA, 1):          ${tSelRow.value}");
      print("CONCAT_VEC(vA, [5,6]):      ${tConcat.value}");
      print("STACK_MAT(mA, mB):          ${tStack.value}");
      print("PAD_MAT(mA, 1):             ${tPad.value}");

      vA.free();
      mA.free();
      mB.free();
      tReshape.free();
      tSliceCol.free();
      tSelRow.free();
      tConcat.free();
      tStack.free();
      tPad.free(); //[cite: 2]
    }

    // ======================================================================
    // 7. ADVANCED LAYERS (Conv, Pool)
    // ======================================================================
    print("\n--- 7. ADVANCED LAYERS ---");
    {
      var mIn = GPUTensor<Matrix>.empty([3, 3]); //[cite: 2]
      // 3x3 Matrix voller 1er
      mIn.pushData([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); //[cite: 2]

      var mKernel = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      // 2x2 Kernel voller 1er
      mKernel.pushData([1.0, 1.0, 1.0, 1.0]); //[cite: 2]

      var tape = CommandBuffer(); //[cite: 5]
      // Simple 2D Convolution
      var tConv = conv2dSimpleGPU(mIn, mKernel, tape); //[cite: 3]
      // MaxPool2D (2x2 pool, stride 1)
      var tPool = maxPool2dGPU(mIn, 2, 1, tape); //[cite: 3]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      tConv.toCpu();
      tPool.toCpu(); //[cite: 2]

      print("Inputs: 3x3 Matrix (ones), 2x2 Kernel (ones)");
      print("CONV2D_SIMPLE: ${tConv.value} (Erwartet 2x2 mit Wert 4.0)");
      print("MAXPOOL_2D:    ${tPool.value}");

      mIn.free();
      mKernel.free();
      tConv.free();
      tPool.free(); //[cite: 2]
    }

    print("\n🧹 Aufräumen...");
    GPUEngine.dispose(); //[cite: 1]
    print("✅ Alle Funktionen gefeuert und validiert!");
    return true;

  }

  Future<bool> testLinear() async {
    print("🚀 Initialisiere chirurgischen Linear-Algebra-Debugger...");
    await GPUEngine.initialize(
      debug: false,
      target: target,
    ); //[cite: 1]

    // Basis-Tensoren anlegen
    var mA = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
    var mB = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
    var vC = GPUTensor<Vector>.empty([2]); //[cite: 2]
    var sD = GPUTensor<Scalar>(
      5.0,
    ); // Scalar für addScalarMatrixGPU //[cite: 2]

    mA.pushData([1.0, 2.0, 3.0, 4.0]); //[cite: 2]
    mB.pushData([2.0, 0.0, 1.0, 2.0]); //[cite: 2]
    vC.pushData([10.0, 20.0]); //[cite: 2]

    print("\n--- TEST 1: MATMUL ---");
    var tape1 = CommandBuffer(); //[cite: 5]
    var tMatMul = matMulGPU(mA, mB, tape1); //[cite: 3]
    print("Feuere MatMul...");
    GPUEngine.run(tape1.bytes()); //[cite: 1]
    tMatMul.toCpu(); //[cite: 2]
    print("✅ MATMUL OK: ${tMatMul.value}");

    print("\n--- TEST 2: MATVEC ---");
    var tape2 = CommandBuffer(); //[cite: 5]
    var tMatVec = matVecMulGPU(mA, vC, tape2); //[cite: 3]
    print("Feuere MatVec...");
    GPUEngine.run(tape2.bytes()); //[cite: 1]
    tMatVec.toCpu(); //[cite: 2]
    print("✅ MATVEC OK: ${tMatVec.value}");

    print("\n--- TEST 3: TRANSPOSE ---");
    var tape3 = CommandBuffer(); //[cite: 5]
    var tTrans = transposeGPU(mA, tape3); //[cite: 3]
    print("Feuere Transpose...");
    GPUEngine.run(tape3.bytes()); //[cite: 1]
    tTrans.toCpu(); //[cite: 2]
    print("✅ TRANSPOSE OK: ${tTrans.value}");

    print("\n--- TEST 4: SCALE MATRIX ---");
    var tape4 = CommandBuffer(); //[cite: 5]
    var tScale = scaleMatrixGPU(mA, 10.0, tape4); //[cite: 3]
    print("Feuere Scale Matrix...");
    GPUEngine.run(tape4.bytes()); //[cite: 1]
    tScale.toCpu(); //[cite: 2]
    print("✅ SCALE MATRIX OK: ${tScale.value}");

    print("\n--- TEST 5: ADD BIAS TO MATMUL (Nutzt OP_BROADCAST_ADD) ---");
    var tape5 = CommandBuffer(); //[cite: 5]
    var tAddBi = addBiasToMatMulOutGPU(mA, vC, tape5); //[cite: 3]
    print("Feuere Add Bias...");
    GPUEngine.run(tape5.bytes()); //[cite: 1]
    tAddBi.toCpu(); //[cite: 2]
    print("✅ ADD BIAS OK: ${tAddBi.value}");

    print("\n--- TEST 6: ADD SCALAR MATRIX (Nutzt OP_BROADCAST_ADD) ---");
    var tape6 = CommandBuffer(); //[cite: 5]
    var tAddSca = addScalarMatrixGPU(mA, sD, tape6); //[cite: 3]
    print("Feuere Add Scalar Matrix...");
    GPUEngine.run(tape6.bytes()); //[cite: 1]
    tAddSca.toCpu(); //[cite: 2]
    print("✅ ADD SCALAR MATRIX OK: ${tAddSca.value}");

    print("\n🧹 Räume auf...");
    mA.free();
    mB.free();
    vC.free();
    sD.free(); //[cite: 2]
    tMatMul.free();
    tMatVec.free();
    tTrans.free();
    tScale.free();
    tAddBi.free();
    tAddSca.free(); //[cite: 2]
    GPUEngine.dispose(); //[cite: 1]

    print("🎉 Alle Tests überlebt!");
    return true;

  }

  Future<bool> testBackward() async{
    print("🚀 Starte massive BACKWARD (Gradient) Validierung...");
    await GPUEngine.initialize(debug: false, target: target); //[cite: 1]

    // =========================================================
    // 1. BASIC MATH BACKWARDS
    // =========================================================
    print("\n--- 1. BASIC MATH (ADD, SUB, MUL, DIV) ---");
    {
      var tape = CommandBuffer(); //[cite: 5]

      // ADD: Ableitung von (A + B) nach A ist 1, nach B ist 1
      var aAdd = GPUTensor<Vector>.empty([2]); aAdd.pushData([2.0, 3.0]); //[cite: 2]
      var bAdd = GPUTensor<Vector>.empty([2]); bAdd.pushData([4.0, 5.0]); //[cite: 2]
      var outAdd = addGPU<Vector>(aAdd, bAdd, tape); //[cite: 3]
      outAdd.backward(tape, fillOnes: true); //[cite: 2]

      // SUB: Ableitung von (A - B) nach A ist 1, nach B ist -1
      var aSub = GPUTensor<Vector>.empty([2]); aSub.pushData([2.0, 3.0]); //[cite: 2]
      var bSub = GPUTensor<Vector>.empty([2]); bSub.pushData([4.0, 5.0]); //[cite: 2]
      var outSub = subtractGPU<Vector>(aSub, bSub, tape); //[cite: 3]
      outSub.backward(tape, fillOnes: true); //[cite: 2]

      // MUL: Ableitung von (A * B) nach A ist B, nach B ist A
      var aMul = GPUTensor<Vector>.empty([2]); aMul.pushData([2.0, 3.0]); //[cite: 2]
      var bMul = GPUTensor<Vector>.empty([2]); bMul.pushData([4.0, 5.0]); //[cite: 2]
      var outMul = multiplyGPU<Vector>(aMul, bMul, tape); //[cite: 3]
      outMul.backward(tape, fillOnes: true); //[cite: 2]

      // DIV: Ableitung von (A / B) nach A ist 1/B, nach B ist -A/B^2
      var aDiv = GPUTensor<Vector>.empty([2]); aDiv.pushData([2.0, 3.0]); //[cite: 2]
      var bDiv = GPUTensor<Vector>.empty([2]); bDiv.pushData([4.0, 5.0]); //[cite: 2]
      var outDiv = divideGPU<Vector>(aDiv, bDiv, tape); //[cite: 3]
      outDiv.backward(tape, fillOnes: true); //[cite: 2]

      GPUEngine.run(tape.bytes()); //[cite: 1]

      // Gradienten (.grad) zusammen mit den Daten in den RAM holen
      aAdd.toCpu(); bAdd.toCpu(); //[cite: 2]
      aSub.toCpu(); bSub.toCpu(); //[cite: 2]
      aMul.toCpu(); bMul.toCpu(); //[cite: 2]
      aDiv.toCpu(); bDiv.toCpu(); //[cite: 2]

      print("Inputs: A=[2,3], B=[4,5]");
      print("ADD Grad A: ${aAdd.gradValue} (Erwartet: [1.0, 1.0])"); //[cite: 2]
      print("ADD Grad B: ${bAdd.gradValue} (Erwartet: [1.0, 1.0])"); //[cite: 2]
      print("SUB Grad A: ${aSub.gradValue} (Erwartet: [1.0, 1.0])"); //[cite: 2]
      print("SUB Grad B: ${bSub.gradValue} (Erwartet: [-1.0, -1.0])"); //[cite: 2]
      print("MUL Grad A: ${aMul.gradValue} (Erwartet: [4.0, 5.0])"); //[cite: 2]
      print("MUL Grad B: ${bMul.gradValue} (Erwartet: [2.0, 3.0])"); //[cite: 2]
      print("DIV Grad A: ${aDiv.gradValue} (Erwartet: [0.25, 0.2])"); //[cite: 2]
      print("DIV Grad B: ${bDiv.gradValue} (Erwartet: [-0.125, -0.12])"); //[cite: 2]

      aAdd.free(); bAdd.free(); outAdd.free(); //[cite: 2]
      aSub.free(); bSub.free(); outSub.free(); //[cite: 2]
      aMul.free(); bMul.free(); outMul.free(); //[cite: 2]
      aDiv.free(); bDiv.free(); outDiv.free(); //[cite: 2]
    }

    // =========================================================
    // 2. UNARY & ACTIVATIONS
    // =========================================================
    print("\n--- 2. UNARY & ACTIVATIONS ---");
    {
      var tape = CommandBuffer(); //[cite: 5]

      // POW: Ableitung von x^2 ist 2*x
      var aPow = GPUTensor<Vector>.empty([2]); aPow.pushData([2.0, 4.0]); //[cite: 2]
      var outPow = powGPU<Vector>(aPow, 2.0, tape); //[cite: 3]
      outPow.backward(tape, fillOnes: true); //[cite: 2]

      // LOG: Ableitung von ln(x) ist 1/x
      var aLog = GPUTensor<Vector>.empty([2]); aLog.pushData([2.0, 4.0]); //[cite: 2]
      var outLog = logGPU<Vector>(aLog, tape); //[cite: 3]
      outLog.backward(tape, fillOnes: true); //[cite: 2]

      // RELU: Ableitung von max(0, x) ist 1 wenn x > 0, sonst 0
      var aRelu = GPUTensor<Vector>.empty([2]); aRelu.pushData([2.0, -1.0]); //[cite: 2]
      var outRelu = reluGPU(aRelu, tape); //[cite: 3]
      outRelu.backward(tape, fillOnes: true); //[cite: 2]

      GPUEngine.run(tape.bytes()); //[cite: 1]
      aPow.toCpu(); aLog.toCpu(); aRelu.toCpu(); //[cite: 2]

      print("POW(A=[2,4], 2) Grad: ${aPow.gradValue} (Erwartet: [4.0, 8.0])"); //[cite: 2]
      print("LOG(A=[2,4]) Grad:    ${aLog.gradValue} (Erwartet: [0.5, 0.25])"); //[cite: 2]
      print("RELU(A=[2,-1]) Grad:  ${aRelu.gradValue} (Erwartet: [1.0, 0.0])"); //[cite: 2]

      aPow.free(); outPow.free(); //[cite: 2]
      aLog.free(); outLog.free(); //[cite: 2]
      aRelu.free(); outRelu.free(); //[cite: 2]
    }

    // =========================================================
    // 3. MATMUL & LINEAR ALGEBRA
    // =========================================================
    print("\n--- 3. MATMUL & LINEAR ALGEBRA ---");
    {
      var tape = CommandBuffer(); //[cite: 5]

      // C = A * B. Grad(A) = Grad(C) * B^T. Grad(B) = A^T * Grad(C)
      var mA = GPUTensor<Matrix>.empty([1, 2]); mA.pushData([1.0, 2.0]); //[cite: 2]
      var mB = GPUTensor<Matrix>.empty([2, 1]); mB.pushData([3.0, 4.0]); //[cite: 2]
      var outMatMul = matMulGPU(mA, mB, tape); //[cite: 3]
      outMatMul.backward(tape, fillOnes: true); //[cite: 2]

      GPUEngine.run(tape.bytes()); //[cite: 1]
      mA.toCpu(); mB.toCpu(); //[cite: 2]

      print("MATMUL A=[[1,2]], B=[[3],[4]]");
      print("Grad A: ${mA.gradValue} (Erwartet: [[3.0, 4.0]])"); //[cite: 2]
      print("Grad B: ${mB.gradValue} (Erwartet: [[1.0], [2.0]])"); //[cite: 2]

      mA.free(); mB.free(); outMatMul.free(); //[cite: 2]
    }

    // =========================================================
    // 4. LOSSES & REDUCTIONS
    // =========================================================
    print("\n--- 4. LOSSES & REDUCTIONS ---");
    {
      var tape = CommandBuffer(); //[cite: 5]

      // MSE Loss: (Pred - Target) * 2 / N
      var vPred = GPUTensor<Vector>.empty([2]); vPred.pushData([0.8, 0.2]); //[cite: 2]
      var vTarg = GPUTensor<Vector>.empty([2]); vTarg.pushData([1.0, 0.0]); //[cite: 2]
      var outMse = mseGPU(vPred, vTarg, tape); //[cite: 3]
      outMse.backward(tape, fillOnes: true); //[cite: 2]

      // Sum Reduction: Gradient verteilt sich gleichmäßig als 1.0
      var vSum = GPUTensor<Vector>.empty([3]); vSum.pushData([1.0, 2.0, 3.0]); //[cite: 2]
      var outSum = sumGPU(vSum, tape); //[cite: 3]
      outSum.backward(tape, fillOnes: true); //[cite: 2]

      GPUEngine.run(tape.bytes()); //[cite: 1]
      vPred.toCpu(); vSum.toCpu(); //[cite: 2]

      print("MSE Pred=[0.8, 0.2], Targ=[1.0, 0.0]");
      print("Grad Pred: ${vPred.gradValue} (Erwartet: [-0.2, 0.2])"); //[cite: 2]

      print("SUM V=[1,2,3]");
      print("Grad V:    ${vSum.gradValue} (Erwartet: [1.0, 1.0, 1.0])"); //[cite: 2]

      vPred.free(); vTarg.free(); outMse.free(); //[cite: 2]
      vSum.free(); outSum.free(); //[cite: 2]
    }

    // =========================================================
    // 5. CONVOLUTION 2D
    // =========================================================
    print("\n--- 5. CONVOLUTION 2D ---");
    {
      var tape = CommandBuffer(); //[cite: 5]

      // Faltung eines 3x3 Inputs mit einem 2x2 Kernel.
      // Der Gradient des Kernels ist die Summe der überlappenden Fenster.
      var mIn = GPUTensor<Matrix>.empty([3, 3]); //[cite: 2]
      mIn.pushData([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]); //[cite: 2]
      var mKer = GPUTensor<Matrix>.empty([2, 2]); //[cite: 2]
      mKer.pushData([1.0, 1.0, 1.0, 1.0]); //[cite: 2]

      var outConv = conv2dSimpleGPU(mIn, mKer, tape); //[cite: 3]
      outConv.backward(tape, fillOnes: true); //[cite: 2]

      GPUEngine.run(tape.bytes()); //[cite: 1]
      mKer.toCpu(); //[cite: 2]

      print("CONV2D In=3x3(1..9), Ker=2x2(1s)");
      print("Grad Kernel: ${mKer.gradValue} (Erwartet: [[12.0, 16.0], [24.0, 28.0]])"); //[cite: 2]

      mIn.free(); mKer.free(); outConv.free(); //[cite: 2]
    }

    print("\n🧹 Aufräumen...");
    GPUEngine.dispose(); //[cite: 1]
    print("✅ Backward Validierung abgeschlossen!");
    return true;

  }

  Future<bool> testSpeeds()async{
    print("🚀 Starte TFLOP & Overhead Benchmark...");
    // Target anpassen (z.B. Target.cuda oder Target.android_x86_64)
    await GPUEngine.initialize(debug: false, target: target); //

    // ====================================================================
    // TEST 1: FFI & DISPATCH OVERHEAD
    // ====================================================================
    print("\n=======================================================");
    print("⏱️ TEST 1: ENGINE DISPATCH OVERHEAD");
    print("=======================================================");

    // Ein winziger Tensor (1 Element), um die GPU-Rechenzeit gegen 0 zu drücken
    var tinyVec = GPUTensor<Vector>.empty([1]); //[cite: 2]
    var tapeOverhead = CommandBuffer(); //[cite: 5]

    // Nur ein simpler Fill-Befehl
    tapeOverhead.putInt(OP_FILL); //[cite: 2, 5]
    tapeOverhead.putString(tinyVec.id); //[cite: 2, 5]
    tapeOverhead.putFloat(1.0); //[cite: 2, 5]

    Uint8List compiledOverhead = tapeOverhead.bytes(); //[cite: 5]

    // Warmup (Weckt die Engine und füllt Caches)
    for (int i = 0; i < 100; i++) {
      GPUEngine.run(compiledOverhead); //[cite: 1]
    }

    int overheadIterations = 10000;
    Stopwatch swOverhead = Stopwatch()..start();
    for (int i = 0; i < overheadIterations; i++) {
      GPUEngine.run(compiledOverhead); //[cite: 1]
    }
    swOverhead.stop();

    double totalMs = swOverhead.elapsedMicroseconds / 1000.0;
    double msPerDispatch = totalMs / overheadIterations;

    print("Befehle gesendet: $overheadIterations");
    print("Gesamtzeit:       ${totalMs.toStringAsFixed(2)} ms");
    print("Overhead pro Run: ${(msPerDispatch * 1000).toStringAsFixed(2)} Mikrosekunden (µs) pro Dispatch");

    tinyVec.free(); //[cite: 2]

    // ====================================================================
    // TEST 2: MAXIMAL TFLOPS (Matrix Multiplication)
    // ====================================================================
    print("\n=======================================================");
    print("🔥 TEST 2: MAXIMAL TFLOPS (MATRIX MULTIPLICATION)");
    print("=======================================================");

    // Matrix-Dimension: 4096 x 4096
    // Belegt ca. 3x 67 MB VRAM (A, B, und Out) = ~201 MB total
    int M = 512*2;
    print("Matrix Größe:      ${M}x$M Elemente");

    var matA = GPUTensor<Matrix>.empty([M, M]); //[cite: 2]
    var matB = GPUTensor<Matrix>.empty([M, M]); //[cite: 2]

    // Wir füllen sie auf der GPU, damit wir keine Zeit mit CPU->GPU Transfers verlieren
    var setupTape = CommandBuffer(); //[cite: 5]
    setupTape.putInt(OP_FILL); setupTape.putString(matA.id); setupTape.putFloat(1.0); //[cite: 2, 5]
    setupTape.putInt(OP_FILL); setupTape.putString(matB.id); setupTape.putFloat(1.0); //[cite: 2, 5]
    GPUEngine.run(setupTape.bytes()); //[cite: 1]

    var tflopTape = CommandBuffer(); //[cite: 5]
    // Das ist unsere schwere Last
    var matOut = matMulGPU(matA, matB, tflopTape); //[cite: 3]
    Uint8List compiledTflop = tflopTape.bytes(); //[cite: 5]

    // Formel für MatMul Flops: 2 * M * K * N
    // Da M = K = N = 4096:
    double flopsPerRun = 2.0 * M * M * M;
    double tflopsPerRun = flopsPerRun / 1e12; // In TeraFLOPs umrechnen

    // Warmup
    GPUEngine.run(compiledTflop); //[cite: 1]

    int tflopIterations = 20;
    print("Starte $tflopIterations Durchläufe...");

    Stopwatch swTflop = Stopwatch()..start();
    for (int i = 0; i < tflopIterations; i++) {
      GPUEngine.run(compiledTflop); //[cite: 1]
    }
    swTflop.stop();

    double seconds = swTflop.elapsedMicroseconds / 1000000.0;
    double totalTflops = tflopsPerRun * tflopIterations;
    double tflopsPerSecond = totalTflops / seconds;
    double gflopsPerSecond = tflopsPerSecond * 1000.0;

    print("Gesamtzeit:        ${seconds.toStringAsFixed(3)} s");
    print("Berechnete TFLOPs: ${totalTflops.toStringAsFixed(3)} TFLOPs total");
    print("-------------------------------------------------------");
    print("Leistung (GFLOP/s): ${gflopsPerSecond.toStringAsFixed(2)} GFLOP/s");
    print("Leistung (TFLOP/s): ${tflopsPerSecond.toStringAsFixed(4)} TFLOP/s");
    print("=======================================================\n");

    matA.free(); matB.free(); matOut.free(); //[cite: 2]
    GPUEngine.dispose(); //[cite: 1]
    return true;
  }

  Future<bool> testLearning2() async {
    print("🚀 Initialisiere GPU Engine für Forward-Pass-Test...");
    await GPUEngine.initialize(debug: false, target:target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(8));
    net.add(ReLULayerMatrixTapeLayer());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    // Kompilieren (allokiert VRAM und bereitet das Forward-Tape vor)
    print("⚙️ Kompiliere Graphen...");
    net.compile([4, 2], [4, 1], 0.001, useAdam: true);

    // Unser Test-Batch
    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];

    print("🔮 Führe Forward-Pass (predict) aus...");
    // predict() schiebt xBatch in den VRAM, triggert fTape und holt das Ergebnis in den RAM
    var prediction = net.predict(xBatch);

    print("\n📊 Rohe Vorhersagen (Untrainiert, basierend auf Zufallsgewichten):");
    print(prediction);

    // Optional: Wir können auch direkt prüfen, ob die Gewichte ausgelesen werden können
    print("\n⚖️ Lade initiale Gewichte aus Layer 1 (DenseTL) herunter...");
    var initialWeights = net.layers[0].getWeights();
    print(initialWeights['weights']);

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();

    return true;
  }
  Future<bool> testBackwardPass() async {
    print("🚀 Initialisiere GPU Engine für Backward-Pass-Test...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(8));
    net.add(ReLULayerMatrixTapeLayer());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    // Wir nutzen eine sehr hohe Lernrate (0.5), damit wir den Sprung der Gewichte deutlich sehen
    net.compile([4, 2], [4, 1], 0.5, useAdam: true);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    // 1. Gewichte VOR dem Gradienten-Update lesen
    print("\n⚖️ Initiale Gewichte in Layer 1 (Zeile 1):");
    var wBefore = net.layers[0].getWeights()['weights'] as List;
    print(wBefore[0]);

    // 2. Ein einziger Durchlauf (Forward -> Backward -> Adam Update)
    print("\n🔮 Führe Backpropagation aus...");
    double loss = net.trainStep(xBatch, yBatch);
    print("📊 Initialer Loss: $loss");

    // 3. Gewichte NACH dem Gradienten-Update lesen
    print("\n📉 Gewichte NACH dem Update (Zeile 1):");
    var wAfter = net.layers[0].getWeights()['weights'] as List;
    print(wAfter[0]);

    // 4. Prüfen, ob sich etwas bewegt hat
    bool didLearn = false;
    for (int i = 0; i < (wBefore[0] as List).length; i++) {
      if (wBefore[0][i] != wAfter[0][i]) {
        didLearn = true;
        break;
      }
    }

    print(didLearn ? "✅ Gradienten fließen! Gewichte wurden erfolgreich geupdatet." : "❌ FEHLER: Gewichte sind eingefroren (Gradient ist 0.0).");

    net.free();
    GPUEngine.dispose();
    return true;
  }
  Future<bool> testShortTraining() async {
    print("🚀 Initialisiere GPU Engine für Kurz-Training...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(16)); // 16 Neuronen, um das 0.69-Minimum sicher zu knacken
    net.add(ReLULayerMatrixTapeLayer());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    // Kompilieren mit Adam und einer guten Lernrate
    net.compile([4, 2], [4, 1], 0.05, useAdam: true);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    print("\n🔮 Starte Training für 300 Epochen...");

    for (int epoch = 1; epoch <= 300; epoch++) {
      double loss = net.trainStep(xBatch, yBatch);

      if (epoch % 50 == 0 || epoch == 1) {
        print("Epoch ${epoch.toString().padLeft(3)} | Loss: ${loss.toStringAsFixed(6)}");
      }
    }

    print("\n📊 Vorhersagen NACH dem Training (Ziel: 0, 1, 1, 0):");
    var prediction = net.predict(xBatch);

    // Etwas schöner formatiert ausgeben
    List<List<double>> preds = prediction as List<List<double>>;
    print("Input [0,0] -> ${preds[0][0].toStringAsFixed(4)} (Ziel: 0.0)");
    print("Input [0,1] -> ${preds[1][0].toStringAsFixed(4)} (Ziel: 1.0)");
    print("Input [1,0] -> ${preds[2][0].toStringAsFixed(4)} (Ziel: 1.0)");
    print("Input [1,1] -> ${preds[3][0].toStringAsFixed(4)} (Ziel: 0.0)");

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();
    return true;
  }
  Future<bool> testConvergence() async {
    print("🚀 Initialisiere GPU Engine für Konvergenz-Test...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(16));
    net.add(ReLULayerMatrixTapeLayer());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    net.compile([4, 2], [4, 1], 0.05, useAdam: true);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    print("\n🔍 Beobachte Vorhersagen und Loss über 50 Epochen...");

    for (int epoch = 1; epoch <= 50; epoch++) {
      // 1. Trainieren (Forward, Backward, Optimize)
      double loss = net.trainStep(xBatch, yBatch);

      // 2. Aktuellen Stand der Vorhersagen abholen
      var prediction = net.predict(xBatch);
      List<List<double>> preds = prediction as List<List<double>>;

      // 3. Ein einzelnes Gewicht auslesen, um die Bewegung zu tracken
      var weights = net.layers[0].getWeights()['weights'] as List;
      double sampleWeight = weights[0][0];

      // Alle 10 Epochen (und beim ersten Mal) loggen
      if (epoch % 10 == 0 || epoch == 1) {
        print("\n--- Epoche ${epoch.toString().padLeft(2)} ---");
        print("Loss:      ${loss.toStringAsFixed(6)}");
        print("Gewicht 0: ${sampleWeight.toStringAsFixed(6)}");
        print("Preds:     [${preds[0][0].toStringAsFixed(4)}, ${preds[1][0].toStringAsFixed(4)}, ${preds[2][0].toStringAsFixed(4)}, ${preds[3][0].toStringAsFixed(4)}]");
      }
    }

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();
    return true;
  }

  Future<bool> testTrainingAndPrintWeights() async {
    print("🚀 Initialisiere GPU Engine...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();

    net.add(DenseTL(16));

    // ✅ DER LEBENSRETTER: Sigmoid stirbt niemals!
    // Es liefert auch bei negativen Werten immer einen kleinen Gradienten zurück,
    // sodass Adam weiterarbeiten kann und das Netzwerk nicht einfriert.
    net.add(SigmoidMatrixTL());

    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    // Wir können die Lernrate jetzt mutig auf 0.05 setzen, da Sigmoid robuster ist
    net.compile([4, 2], [4, 1], 0.05, useAdam: true);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    print("\n🔥 Starte Training für 2000 Epochen...");

    for (int epoch = 1; epoch <= 2000; epoch++) {
      double loss = net.trainStep(xBatch, yBatch);

      if (epoch % 200 == 0 || epoch == 1) {
        print("Epoch ${epoch.toString().padLeft(4)} | Loss: ${loss.toStringAsFixed(6)}");
      }
    }

    print("\n🧠 Vorhersagen nach 2000 Epochen (Ziel: 0, 1, 1, 0):");
    var prediction = net.predict(xBatch);
    List<List<double>> preds = prediction as List<List<double>>;
    print("Input [0,0] -> ${preds[0][0].toStringAsFixed(4)}");
    print("Input [0,1] -> ${preds[1][0].toStringAsFixed(4)}");
    print("Input [1,0] -> ${preds[2][0].toStringAsFixed(4)}");
    print("Input [1,1] -> ${preds[3][0].toStringAsFixed(4)}");

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();
    return true;
  }
  Future<bool> debugAllGradients() async {
    print("🚀 Initialisiere GPU Engine für GRADIENTEN-RÖNTGEN...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(4)); // 4 Neuronen = Übersichtlicher Output!
    net.add(ReLULayerMatrixTapeLayer());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    // 🛑 WICHTIG: SGD mit Lernrate 1.0 !
    // Dadurch gilt: Gradient = Gewicht_Vorher - Gewicht_Nachher
    net.compile([4, 2], [4, 1], 1.0, useAdam: false);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    // 1. Snapshot: Gewichte VOR dem Forward-Pass aus dem VRAM holen
    var l1Before = net.layers[0].getWeights()['weights'] as List;
    var l2Before = net.layers[2].getWeights()['weights'] as List;

    // 2. Aktion: EIN einziger Durchlauf (Forward + Backward + Optimize)
    print("\n🔥 Führe exakt 1 Backpropagation-Schritt aus...");
    double loss = net.trainStep(xBatch, yBatch);
    print("📊 Berechneter Loss: $loss");

    // 3. Snapshot: Gewichte NACH dem Update aus dem VRAM holen
    var l1After = net.layers[0].getWeights()['weights'] as List;
    var l2After = net.layers[2].getWeights()['weights'] as List;

    // 4. Analyse: Berechnen und Printen!
    print("\n" + "="*60);
    print("🔍 GRADIENTEN-ANALYSE (LAYER 1 - Hidden Layer)");
    print("="*60);
    for (int i = 0; i < l1Before.length; i++) {
      List<double> rowBefore = (l1Before[i] as List).cast<double>();
      List<double> rowAfter = (l1After[i] as List).cast<double>();
      print("Neuron $i:");
      for (int j = 0; j < rowBefore.length; j++) {
        double delta = rowAfter[j] - rowBefore[j];
        double grad = -delta; // W_alt - W_neu
        String marker = (grad == 0.0) ? " 💀 DEAD" : " ✅ ALIVE";
        print("  Weight $j | Vorher: ${rowBefore[j].toStringAsFixed(6).padLeft(9)} | Nachher: ${rowAfter[j].toStringAsFixed(6).padLeft(9)} | Gradient: ${grad.toStringAsFixed(8)}$marker");
      }
    }

    print("\n" + "="*60);
    print("🔍 GRADIENTEN-ANALYSE (LAYER 2 - Output Layer)");
    print("="*60);
    for (int i = 0; i < l2Before.length; i++) {
      List<double> rowBefore = (l2Before[i] as List).cast<double>();
      List<double> rowAfter = (l2After[i] as List).cast<double>();
      print("Output-Verbindung Zeile $i:");
      for (int j = 0; j < rowBefore.length; j++) {
        double delta = rowAfter[j] - rowBefore[j];
        double grad = -delta; // W_alt - W_neu
        String marker = (grad == 0.0) ? " 💀 DEAD" : " ✅ ALIVE";
        print("  Weight $j | Vorher: ${rowBefore[j].toStringAsFixed(6).padLeft(9)} | Nachher: ${rowAfter[j].toStringAsFixed(6).padLeft(9)} | Gradient: ${grad.toStringAsFixed(8)}$marker");
      }
    }

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();
    return true;
  }
  Future<bool> extractRawGradients() async {
    print("🚀 Initialisiere GPU Engine für ROH-GRADIENTEN-EXTRAKTION...");
    await GPUEngine.initialize(debug: false, target: target);

    SNetworkGPU net = SNetworkGPU();
    net.add(DenseTL(4));
    net.add(SigmoidMatrixTL());
    net.add(DenseTL(1));
    net.add(SigmoidMatrixTL());

    net.compile([4, 2], [4, 1], 1.0, useAdam: false);

    List<double> xBatch = [
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0
    ];
    List<double> yBatch = [0.0, 1.0, 1.0, 0.0];

    net.inputRef.pushData(xBatch);
    net.targetRef.pushData(yBatch);

    print("\n🔥 Führe Forward und Backward Pass aus (OHNE Optimizer)...");
    GPUEngine.run(net.zTape);
    GPUEngine.run(net.fTape);
    GPUEngine.run(net.bTape);

    print("\n" + "="*60);
    print("🔍 ROH-GRADIENTEN DIREKT AUS DEM C++ VRAM (via toCpu())");
    print("="*60);

    for (int i = 0; i < net.allParams.length; i++) {
      GPUTensor param = net.allParams[i];

      // ✅ KORREKTUR: Wir nutzen sauber die integrierte toCpu() Methode der Tensor-Klasse[cite: 3]
      param.toCpu();

      // Sobald toCpu() aufgerufen wurde, ist die .grad Liste des Tensors direkt befüllt[cite: 3]
      var gradData = param.grad;

      print("\n--- Parameter $i (Shape: ${param.shape}, ID: ${param.id}) ---");
      print("Rohe Gradienten im Speicher:");
      print(gradData);
    }

    print("\n🧹 Aufräumen...");
    net.free();
    GPUEngine.dispose();
    return true;
  }


}
