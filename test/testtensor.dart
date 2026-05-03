import 'dart:math' as math;

import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/gpu_version/ffi/tapeDecoder.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/tensor_math_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';

void main() {
  print("[MAIN] Initializing CUDA Engine...");
  CudaEngine.initialize(debug: true);
  print("[MAIN] Engine Initialized Successfully.\n");

  testAbs();
  testSqrt();
  testLog();
  testPow();
  testClamp();

  print("\n[MAIN] Disposing CUDA Engine...");
  CudaEngine.dispose();
  print("[MAIN] Shutdown Complete.");
}

void testAbs() {
  print("==========================================");
  print("[ABS] Starting Test...");
  CommandBuffer fTape = CommandBuffer();
  CommandBuffer bTape = CommandBuffer();

  List<double> dataIn = <double>[-3.0, -1.0, 0.0, 1.0, 3.0];
  print("[ABS] Allocating Tensor...");
  GPUTensor<Vector> tIn = GPUTensor<Vector>(dataIn);

  print("[ABS] Building Forward Tape...");
  GPUTensor<Vector> tOut = absGPU<Vector>(tIn, fTape);

  print("[ABS] Building Backward Tape...");
  tOut.backward(bTape, fillOnes: true);

  print("[ABS] Pushing Forward Tape to Engine (Bytes: ${fTape.bytes().length})...");
  CudaEngine.run(fTape.bytes());
  print("[ABS] Forward Tape Executed.");

  print("[ABS] Pushing Backward Tape to Engine (Bytes: ${bTape.bytes().length})...");
  TapeDecoder(bTape.bytes()).decode();

  CudaEngine.run(bTape.bytes());
  print("[ABS] Backward Tape Executed.");

  print("[ABS] Retrieving Memory to CPU...");
  tIn.toCpu();
  tOut.toCpu();

  print("--- ABS Results ---");
  for (int i = 0; i < dataIn.length; i = i + 1) {
    print("In: " + tIn.data[i].toString() +
        " | Out: " + tOut.data[i].toString() +
        " | Grad: " + tIn.grad[i].toString());
  }

  print("[ABS] Freeing VRAM...");
  tIn.free();
  tOut.free();
  print("[ABS] Test Complete.\n");
}

void testSqrt() {
  print("==========================================");
  print("[SQRT] Starting Test...");
  CommandBuffer fTape = CommandBuffer();
  CommandBuffer bTape = CommandBuffer();

  List<double> dataIn = <double>[1.0, 4.0, 9.0, 16.0];
  print("[SQRT] Allocating Tensor...");
  GPUTensor<Vector> tIn = GPUTensor<Vector>(dataIn);

  print("[SQRT] Building Forward Tape...");
  GPUTensor<Vector> tOut = sqrtGPU<Vector>(tIn, fTape);

  print("[SQRT] Building Backward Tape...");
  tOut.backward(bTape, fillOnes: true);

  print("[SQRT] Pushing Forward Tape to Engine (Bytes: ${fTape.bytes().length})...");
  CudaEngine.run(fTape.bytes());
  print("[SQRT] Forward Tape Executed.");

  print("[SQRT] Pushing Backward Tape to Engine (Bytes: ${bTape.bytes().length})...");
  CudaEngine.run(bTape.bytes());
  print("[SQRT] Backward Tape Executed.");

  print("[SQRT] Retrieving Memory to CPU...");
  tIn.toCpu();
  tOut.toCpu();

  print("--- SQRT Results ---");
  for (int i = 0; i < dataIn.length; i = i + 1) {
    print("In: " + tIn.data[i].toString() +
        " | Out: " + tOut.data[i].toString() +
        " | Grad: " + tIn.grad[i].toString());
  }

  print("[SQRT] Freeing VRAM...");
  tIn.free();
  tOut.free();
  print("[SQRT] Test Complete.\n");
}

void testLog() {
  print("==========================================");
  print("[LOG] Starting Test...");
  CommandBuffer fTape = CommandBuffer();
  CommandBuffer bTape = CommandBuffer();

  List<double> dataIn = <double>[1.0, math.e, math.pow(math.e, 2).toDouble()];
  print("[LOG] Allocating Tensor...");
  GPUTensor<Vector> tIn = GPUTensor<Vector>(dataIn);

  print("[LOG] Building Forward Tape...");
  GPUTensor<Vector> tOut = logGPU<Vector>(tIn, fTape);

  print("[LOG] Building Backward Tape...");
  tOut.backward(bTape, fillOnes: true);

  print("[LOG] Pushing Forward Tape to Engine (Bytes: ${fTape.bytes().length})...");
  CudaEngine.run(fTape.bytes());
  print("[LOG] Forward Tape Executed.");

  print("[LOG] Pushing Backward Tape to Engine (Bytes: ${bTape.bytes().length})...");
  CudaEngine.run(bTape.bytes());
  print("[LOG] Backward Tape Executed.");

  print("[LOG] Retrieving Memory to CPU...");
  tIn.toCpu();
  tOut.toCpu();

  print("--- LOG Results ---");
  for (int i = 0; i < dataIn.length; i = i + 1) {
    print("In: " + tIn.data[i].toString() +
        " | Out: " + tOut.data[i].toString() +
        " | Grad: " + tIn.grad[i].toString());
  }

  print("[LOG] Freeing VRAM...");
  tIn.free();
  tOut.free();
  print("[LOG] Test Complete.\n");
}

void testPow() {
  print("==========================================");
  print("[POW] Starting Test...");
  CommandBuffer fTape = CommandBuffer();
  CommandBuffer bTape = CommandBuffer();

  List<double> dataIn = <double>[1.0, 2.0, 3.0, 4.0];
  double exponent = 3.0;
  print("[POW] Allocating Tensor...");
  GPUTensor<Vector> tIn = GPUTensor<Vector>(dataIn);

  print("[POW] Building Forward Tape...");
  GPUTensor<Vector> tOut = powGPU<Vector>(tIn, exponent, fTape);

  print("[POW] Building Backward Tape...");
  tOut.backward(bTape, fillOnes: true);

  print("[POW] Pushing Forward Tape to Engine (Bytes: ${fTape.bytes().length})...");
  CudaEngine.run(fTape.bytes());
  print("[POW] Forward Tape Executed.");

  print("[POW] Pushing Backward Tape to Engine (Bytes: ${bTape.bytes().length})...");
  CudaEngine.run(bTape.bytes());
  print("[POW] Backward Tape Executed.");

  print("[POW] Retrieving Memory to CPU...");
  tIn.toCpu();
  tOut.toCpu();

  print("--- POW Results (Exponent: " + exponent.toString() + ") ---");
  for (int i = 0; i < dataIn.length; i = i + 1) {
    print("In: " + tIn.data[i].toString() +
        " | Out: " + tOut.data[i].toString() +
        " | Grad: " + tIn.grad[i].toString());
  }

  print("[POW] Freeing VRAM...");
  tIn.free();
  tOut.free();
  print("[POW] Test Complete.\n");
}

void testClamp() {
  print("==========================================");
  print("[CLAMP] Starting Test...");
  CommandBuffer fTape = CommandBuffer();
  CommandBuffer bTape = CommandBuffer();

  List<double> dataIn = <double>[-5.0, -1.0, 2.0, 6.0, 10.0];
  double minVal = 0.0;
  double maxVal = 5.0;
  print("[CLAMP] Allocating Tensor...");
  GPUTensor<Vector> tIn = GPUTensor<Vector>(dataIn);

  print("[CLAMP] Building Forward Tape...");
  GPUTensor<Vector> tOut = clampGPU<Vector>(tIn, minVal, maxVal, fTape);

  print("[CLAMP] Building Backward Tape...");
  tOut.backward(bTape, fillOnes: true);

  print("[CLAMP] Pushing Forward Tape to Engine (Bytes: ${fTape.bytes().length})...");
  CudaEngine.run(fTape.bytes());
  print("[CLAMP] Forward Tape Executed.");

  print("[CLAMP] Pushing Backward Tape to Engine (Bytes: ${bTape.bytes().length})...");
  CudaEngine.run(bTape.bytes());
  print("[CLAMP] Backward Tape Executed.");

  print("[CLAMP] Retrieving Memory to CPU...");
  tIn.toCpu();
  tOut.toCpu();

  print("--- CLAMP Results (Min: " + minVal.toString() + " | Max: " + maxVal.toString() + ") ---");
  for (int i = 0; i < dataIn.length; i = i + 1) {
    print("In: " + tIn.data[i].toString() +
        " | Out: " + tOut.data[i].toString() +
        " | Grad: " + tIn.grad[i].toString());
  }

  print("[CLAMP] Freeing VRAM...");
  tIn.free();
  tOut.free();
  print("[CLAMP] Test Complete.\n");
}