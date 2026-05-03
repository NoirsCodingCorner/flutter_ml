import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_ml/full_library.dart';


void main() {
  CudaEngine.initialize(debug: false);

  int seqLength = 4;
  int dModel = 4;
  int numHeads = 1;
  int dff = 8;

  double learningRate = 0.05;
  int epochs = 5000;

  print('================================================================================');
  print('                   TINY TRANSFORMER OVERFIT TEST (MEMORY LEAK FIXED)            ');
  print('================================================================================');
  print('Config: SeqLen=$seqLength, dModel=$dModel, Heads=$numHeads, DFF=$dff');
  print('Input: Pure Zeros (Forces network to rely EXCLUSIVELY on Positional Encoding)');
  print('--------------------------------------------------------------------------------');

  List<List<double>> hInput = <List<double>>[];
  for (int i = 0; i < seqLength; i = i + 1) {
    List<double> row = <double>[];
    for (int j = 0; j < dModel; j = j + 1) {
      row.add(0.0);
    }
    hInput.add(row);
  }

  List<List<double>> hTarget = <List<double>>[];
  hTarget.add(<double>[ 1.0, -1.0,  1.0, -1.0]);
  hTarget.add(<double>[-1.0,  1.0, -1.0,  1.0]);
  hTarget.add(<double>[ 1.0,  1.0, -1.0, -1.0]);
  hTarget.add(<double>[-1.0, -1.0,  1.0,  1.0]);

  GPUTensor<Matrix> input = GPUTensor<Matrix>(hInput);
  GPUTensor<Matrix> target = GPUTensor<Matrix>(hTarget);

  PositionalEncodingTL peLayer = PositionalEncodingTL(seqLength, dModel);
  TransformerEncoderBlockTapeLayer block1 = TransformerEncoderBlockTapeLayer(dModel, numHeads, dff);

  peLayer.build(input);
  block1.build(input);

  List<GPUTensor> allParams = <GPUTensor>[];
  List<GPUTensor> b1Params = block1.parameters;
  for (int i = 0; i < b1Params.length; i = i + 1) {
    allParams.add(b1Params[i]);
  }

  CommandBuffer fTape = CommandBuffer();
  List<GPUTensor> intermediates = <GPUTensor>[];

  GPUTensor<Matrix> peOut = peLayer.forward(input, fTape, intermediates) as GPUTensor<Matrix>;
  GPUTensor<Matrix> out1 = block1.forward(peOut, fTape, intermediates) as GPUTensor<Matrix>;

  GPUTensor<Scalar> loss = GPUTensor<Scalar>(0.0);

  fTape.putInt(OP_MSE_LOSS_FORWARD);
  fTape.putString(out1.id);
  fTape.putString(target.id);
  fTape.putString(loss.id);

  loss.creator = GPUNode(
    <GPUTensor>[out1, target],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      bTape.putString(out1.id + '_grad');
      bTape.putString(out1.id);
      bTape.putString(target.id);
      bTape.putString(loss.id + '_grad');
    },
    opName: 'mse_loss_manual',
  );

  Uint8List forwardBytes = fTape.bytes();

  CommandBuffer bTape = CommandBuffer();
  SGDGPU optimizer = SGDGPU(allParams, learningRate);

  optimizer.zeroGrad(bTape);
  for (int i = 0; i < intermediates.length; i = i + 1) {
    bTape.putInt(OP_ZERO_GRAD);
    bTape.putString(intermediates[i].id + '_grad');
  }
  bTape.putInt(OP_ZERO_GRAD);
  bTape.putString(input.id + '_grad');
  bTape.putInt(OP_ZERO_GRAD);
  bTape.putString(out1.id + '_grad');

  bTape.putInt(OP_FILL);
  bTape.putString(loss.id + '_grad');
  bTape.putFloat(1.0);

  loss.backward(bTape);

  for (int i = 0; i < allParams.length; i = i + 1) {
    bTape.putInt(OP_CLIP_GRAD_VALUE);
    bTape.putString(allParams[i].id + '_grad');
    bTape.putFloat(1.0);
  }

  Uint8List backwardBytes = bTape.bytes();

  CommandBuffer oTape = CommandBuffer();
  optimizer.step(oTape);
  Uint8List optimizeBytes = oTape.bytes();

  for (int epoch = 1; epoch <= epochs; epoch = epoch + 1) {
    CudaEngine.run(forwardBytes);
    CudaEngine.run(backwardBytes);
    CudaEngine.run(optimizeBytes);

    if (epoch % 50 == 0) {
      loss.toCpu();
      double currentLoss = loss.value;

      String sEpoch = epoch.toString().padRight(4);
      String sLoss = currentLoss.toStringAsFixed(6);

      print('Epoch $sEpoch | Loss: $sLoss');
    }
  }

  print('--------------------------------------------------------------------------------');
  print('Training Complete. Verifying final pattern output...');

  out1.toCpu();
  List<List<double>> finalOut = out1.value;

  print('Row 0 Target:  [1.0, -1.0,  1.0, -1.0] -> Output: ${finalOut[0]}');
  print('Row 1 Target: [-1.0,  1.0, -1.0,  1.0] -> Output: ${finalOut[1]}');
  print('Row 2 Target:  [1.0,  1.0, -1.0, -1.0] -> Output: ${finalOut[2]}');
  print('Row 3 Target: [-1.0, -1.0,  1.0,  1.0] -> Output: ${finalOut[3]}');

  peLayer.free();
  block1.free();
  input.free();
  target.free();
  loss.free();
  out1.free();
  for (int i = 0; i < intermediates.length; i = i + 1) {
    intermediates[i].free();
  }
}