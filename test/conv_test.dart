// AdamGPU isn't re-exported from optimizer.dart yet (only SGD is), so import it directly.

import 'package:flutter_ml/full_library.dart';

void main() async {
  // Swap for Target.android_arm64 / Target.android_x86_64 when running on device.
  await GPUEngine.initialize(target: Target.cuda);

  // A tiny 1-channel 8x8 "image" and a 2-class target. Both are fixed, so this is
  // deliberately an overfit-one-example sanity check: if it's wired up right, the
  // loss should drop toward 0 over the 50 steps below.
  GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>.empty(<int>[1, 8, 8]);
  GPUTensor<Matrix> target = GPUTensor<Matrix>.empty(<int>[1, 2]);

  SeqModel<Tensor3D, Matrix> model = SeqModel<Tensor3D, Matrix>(
    <TapeLayer>[
      Conv2DTL(4, 3), // [1,8,8]  -> [4,6,6]   (valid padding, 3x3 kernel)
      FlattenTL(), // [4,6,6]  -> [1,144]
      ReLULayerMatrixTapeLayer(), // [1,144]  -> [1,144]   (no Tensor3D relu layer exists yet,
      DenseTL(
          2), // [1,144]  -> [1,2]     so the nonlinearity sits here instead)
    ],
    input,
    target: target,
    lossFunction: mseMatrixGPU,
    optimizerBuilder: (List<GPUTensor> params) => AdamGPU(params, 0.01),
  );

  model.compile();

  List<double> imageData = _checkerboard(8, 8);
  List<double> targetData = <double>[1.0, 0.0];

  for (int step = 0; step < 50; step = step + 1) {
    model.runTraining(inputData: imageData, targetData: targetData);

    if (step % 10 == 0) {
      model.loss!.toCpu();
      print('step $step  loss=${model.loss!.value}');
    }
  }

  model.free();
  GPUEngine.dispose();
}

/// Deterministic 0/1 checkerboard so the example needs no random seed to be reproducible.
List<double> _checkerboard(int height, int width) {
  List<double> pixels = <double>[];
  for (int r = 0; r < height; r = r + 1) {
    for (int c = 0; c < width; c = c + 1) {
      pixels.add(((r + c) % 2).toDouble());
    }
  }
  return pixels;
}
