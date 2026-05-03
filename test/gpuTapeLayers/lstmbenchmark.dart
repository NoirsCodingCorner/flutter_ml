import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/gpu_version/ffi/OpCodes.dart';
import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/gpu_version/optimizer/SGD.dart';
import 'package:flutter_ml/gpu_version/tapelayertypes/LSTMTapeLayer.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';


void main() {
  CudaEngine.initialize(debug: false);
  Random random = Random();

  int seqLength = 128;
  int inputFeatures = 256;
  int hiddenSize = 512;
  double learningRate = 0.01;

  print('================================================================================');
  print('               GIANT NATIVE GPU LSTM MANUAL TRAINING                            ');
  print('================================================================================');
  print('Config: SeqLen=$seqLength, Features=$inputFeatures, Hidden=$hiddenSize');
  print('Clip: 1.0   |   LR: $learningRate   |   Loss: MSE');
  print('--------------------------------------------------------------------------------');

  // 1. Generate Dummy Input (Sequence)
  List<List<double>> hInput = <List<double>>[];
  for (int i = 0; i < seqLength; i = i + 1) {
    List<double> row = <double>[];
    for (int j = 0; j < inputFeatures; j = j + 1) {
      row.add((random.nextDouble() * 2.0) - 1.0);
    }
    hInput.add(row);
  }

  // 2. Generate Dummy Target
  List<double> hTarget = <double>[];
  for (int i = 0; i < hiddenSize; i = i + 1) {
    hTarget.add(0.5);
  }

  GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);
  GPUTensor<Vector> target = GPUTensor<Vector>(hTarget);

  // 3. Build Layer
  LSTMTL lstmLayer = LSTMTL(hiddenSize, gradClipValue: 1.0);
  lstmLayer.build(input);

  print('Compiling Tapes manually...');

  // ===================================================================
  // FORWARD TAPE
  // ===================================================================
  CommandBuffer fTape = CommandBuffer();
  List<GPUTensor> intermediates = <GPUTensor>[];

  GPUTensor<Vector> output = lstmLayer.forward(input, fTape, intermediates) as GPUTensor<Vector>;

  // Allocate Scalar Loss
  GPUTensor<Scalar> loss = GPUTensor<Scalar>(0.0);

  // Write Native MSE Loss Forward
  fTape.putInt(OP_MSE_LOSS_FORWARD);
  fTape.putString(output.id);
  fTape.putString(target.id);
  fTape.putString(loss.id);

  // Link the backward graph from the Loss back to the LSTM Output
  loss.creator = GPUNode(
    <GPUTensor>[output, target],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      bTape.putString(loss.id + '_grad');
      bTape.putString(output.id);
      bTape.putString(target.id);
      bTape.putString(output.id + '_grad');
    },
    opName: 'mse_loss_manual',
  );

  Uint8List forwardBytes = fTape.bytes();

  // ===================================================================
  // BACKWARD TAPE
  // ===================================================================
  CommandBuffer bTape = CommandBuffer();
  SGDGPU optimizer = SGDGPU(lstmLayer.parameters, learningRate);

  // Zero out all gradients
  optimizer.zeroGrad(bTape);
  for (int i = 0; i < intermediates.length; i = i + 1) {
    bTape.putInt(OP_ZERO_GRAD);
    bTape.putString(intermediates[i].id + '_grad');
  }
  bTape.putInt(OP_ZERO_GRAD);
  bTape.putString(input.id + '_grad');
  bTape.putInt(OP_ZERO_GRAD);
  bTape.putString(output.id + '_grad');

  // ⚡ CRITICAL: Inject 1.0 into the loss gradient to kickstart the chain rule
  bTape.putInt(OP_FILL);
  bTape.putString(loss.id + '_grad');
  bTape.putFloat(1.0);

  // Recursively writes all OP_..._BACKWARD instructions to the tape
  loss.backward(bTape);

  Uint8List backwardBytes = bTape.bytes();

  // ===================================================================
  // OPTIMIZER TAPE
  // ===================================================================
  CommandBuffer oTape = CommandBuffer();
  optimizer.step(oTape);
  Uint8List optimizeBytes = oTape.bytes();

  print('Tapes Compiled. Starting 100 Epochs...');
  print('--------------------------------------------------------------------------------');

  // ===================================================================
  // EXECUTION LOOP
  // ===================================================================
  int epochs = 100;
  Stopwatch sw = Stopwatch();

  for (int epoch = 1; epoch <= epochs; epoch = epoch + 1) {
    sw.reset();
    sw.start();

    // 100% Native execution, zero Dart overhead during the mathematical passes
    CudaEngine.run(forwardBytes);
    CudaEngine.run(backwardBytes);
    CudaEngine.run(optimizeBytes);

    sw.stop();

    // Download just the single loss float to monitor progress
    loss.toCpu();
    double currentLoss = loss.value;

    String sEpoch = epoch.toString().padRight(4);
    String sLoss = currentLoss.toStringAsFixed(6).padRight(12);
    String sTime = sw.elapsedMilliseconds.toString();

    print('Epoch $sEpoch | Loss: $sLoss | Time: $sTime ms');
  }

  print('--------------------------------------------------------------------------------');
  print('Training Complete.');

  // Clean up VRAM
  lstmLayer.free();
  input.free();
  target.free();
  loss.free();
  output.free();
  for (int i = 0; i < intermediates.length; i = i + 1) {
    intermediates[i].free();
  }
}