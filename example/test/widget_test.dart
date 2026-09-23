// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_ml/main.dart';

import 'package:flutter_ml/full_library.dart';

void testVecFromVecVec(
  Tensor<Vec> Function(Tensor<Vec>, Tensor<Vec>) cpuMethod,
  GPUTensor<Vec> Function(GPUTensor<Vec>, GPUTensor<Vec>, CommandBuffer)
  gpuMethod,
) {
  GPUTensor<Vec> AG = GPUTensor<Vec>(<double>[1.0, 2.0, 3.0, 4.0]);
  GPUTensor<Vec> BG = GPUTensor<Vec>(<double>[1.0, 2.0, -1.0, 0.0]);

  Tensor<Vec> AC = Tensor<Vec>(<double>[1.0, 2.0, 3.0, 4.0]);
  Tensor<Vec> BC = Tensor<Vec>(<double>[1.0, 2.0, -1.0, 0.0]);

  Tensor<Vec> resultCPU = cpuMethod(AC, BC);

  CommandBuffer buffer = CommandBuffer();
  GPUTensor<Vec> resultGPU = gpuMethod(AG, BG, buffer);

  GPUEngine.run(buffer.bytes());
  resultGPU.toCpu();

  double epsilon = 1e-5;

  for (int i = 0; i < resultCPU.value.length; i++) {
    expect(resultGPU.value[i], closeTo(resultCPU.value[i], epsilon));
  }
}

void main() {
  // setUpAll runs exactly once before any tests start.
  // This guarantees the CUDA FFI binding is ready.
  setUpAll(() async {
    await GPUEngine.initialize(target: Target.cuda);
  });

  test('Add', () {
    testVecFromVecVec(
            (Tensor<Vec> a, Tensor<Vec> b) => addVector(a, b),
            (GPUTensor<Vec> a, GPUTensor<Vec> b, CommandBuffer buf) => addVectorGPU(a, b, buf)
    );
  });
  test('Multiply', () {
    testVecFromVecVec(
            (Tensor<Vec> a, Tensor<Vec> b) => elementWiseMultiply(a, b),
            (GPUTensor<Vec> a, GPUTensor<Vec> b, CommandBuffer buf) => multiplyGPU(a, b, buf)
    );
  });

}
