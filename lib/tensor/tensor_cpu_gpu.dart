import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'tensor.dart';
import 'tensor_gpu.dart';

import '../gpu_version/ffi/OpCodes.dart';
import '../gpu_version/ffi/commandBuffer.dart';
import '../gpu_version/ffi/cudaEngine.dart';


Tensor<T> encapsulateGPUGraph<T>(
    List<Tensor> cpuDynamicInputs,
    List<GPUTensor> gpuDynamicInputs,
    List<Tensor> cpuStaticParams,
    List<GPUTensor> gpuStaticParams,
    GPUTensor gpuOutput,
    T initialOutputValue,
    Uint8List forwardTape,
    Uint8List backwardTape
    ) {

  for (int i = 0; i < cpuDynamicInputs.length; i = i + 1) {
    CudaEngine.load(gpuDynamicInputs[i].id, cpuDynamicInputs[i].dataPtr, gpuDynamicInputs[i].shape);
  }

  CudaEngine.run(forwardTape);

  Tensor<T> out = Tensor<T>(initialOutputValue);
  CudaEngine.retrieve(gpuOutput.id, out.dataPtr);

  List<Tensor> allCpuNodes = [...cpuDynamicInputs, ...cpuStaticParams];
  List<GPUTensor> allGpuNodes = [...gpuDynamicInputs, ...gpuStaticParams];

  out.creator = Node(
      allCpuNodes,
          () {
        // Step 1: Zero the GPU gradients
        CommandBuffer zeroTape = CommandBuffer();
        for (int i = 0; i < allGpuNodes.length; i = i + 1) {
          zeroTape.putInt(OP_ZERO_GRAD);
          zeroTape.putString('${allGpuNodes[i].id}_grad');
        }

        // Fixed: Use .bytes() instead of .buffer
        CudaEngine.run(zeroTape.bytes());

        // Step 2: Push the "starting" gradient for the output back to GPU
        CudaEngine.load('${gpuOutput.id}_grad', out.gradPtr, gpuOutput.shape);

        // Step 3: Run the GPU backward tape
        CudaEngine.run(backwardTape);

        // Step 4: Pull gradients back to CPU and add them
        for (int i = 0; i < allCpuNodes.length; i = i + 1) {
          int numElements = 1;
          List<int> sList = allGpuNodes[i].shape;
          for (int s = 0; s < sList.length; s = s + 1) {
            numElements = numElements * sList[s];
          }

          Pointer<Float> tempGradPtr = calloc<Float>(numElements);
          CudaEngine.retrieve('${allGpuNodes[i].id}_grad', tempGradPtr);
          Float32List tempGradView = tempGradPtr.asTypedList(numElements);

          for (int k = 0; k < numElements; k = k + 1) {
            allCpuNodes[i].grad[k] = allCpuNodes[i].grad[k] + tempGradView[k];
          }

          calloc.free(tempGradPtr);
        }
      },
      opName: 'gpu_graph_block'
  );

  return out;
}

/// Helper to generate dummy nested lists so we can initialize the CPU Tensor
/// without modifying the existing Tensor constructor.
dynamic _createDummy(List<int> shape) {
  if (shape.isEmpty) return 0.0;
  if (shape.length == 1) return List<double>.filled(shape[0], 0.0);
  if (shape.length == 2) return List.generate(shape[0], (_) => List<double>.filled(shape[1], 0.0));
  if (shape.length == 3) return List.generate(shape[0], (_) => List.generate(shape[1], (_) => List<double>.filled(shape[2], 0.0)));
  throw Exception("Shape > 3D not supported for dummy creation");
}

/// Wraps an entire compiled GPU tape into a single CPU Tensor Autograd node.
Tensor<T> executeGPUGraph<T>(
    List<Tensor> cpuInputs,
    List<GPUTensor> gpuInputs,
    GPUTensor gpuOutput,
    CommandBuffer forwardTape,
    CommandBuffer backwardTape, {
      String opName = 'gpu_subgraph',
    }) {
  if (cpuInputs.length != gpuInputs.length) {
    throw Exception("CPU and GPU input lists must be the same length.");
  }

  // 1. Push CPU Data to GPU (Zero-Copy via FFI pointers)
  for (int i = 0; i < cpuInputs.length; i = i + 1) {
    CudaEngine.load(gpuInputs[i].id, cpuInputs[i].dataPtr, gpuInputs[i].shape);
  }

  // 2. Execute the pre-compiled Forward Tape
  CudaEngine.run(forwardTape.bytes());

  // 3. Create the output CPU Tensor & Pull Data
  dynamic dummyVal = _createDummy(gpuOutput.shape);
  Tensor<T> out = Tensor<T>(dummyVal);
  CudaEngine.retrieve(gpuOutput.id, out.dataPtr);

  // 4. Attach the CPU Autograd Node
  out.creator = Node(
    cpuInputs,
        () {
      // A. Zero out GPU gradients for the inputs to prevent accumulation across epochs
      CommandBuffer zeroTape = CommandBuffer();
      for (int i = 0; i < gpuInputs.length; i = i + 1) {
        zeroTape.putInt(OP_ZERO_GRAD);
        zeroTape.putString('${gpuInputs[i].id}_grad');
      }
      CudaEngine.run(zeroTape.bytes());

      // B. Push the accumulated CPU output gradient into the GPU graph
      CudaEngine.load('${gpuOutput.id}_grad', out.gradPtr, gpuOutput.shape);

      // C. Execute the pre-compiled Backward Tape
      CudaEngine.run(backwardTape.bytes());

      // D. Pull GPU input gradients back to the CPU and accumulate them natively
      for (int i = 0; i < cpuInputs.length; i = i + 1) {
        int numElements = 1;
        List<int> sList = gpuInputs[i].shape;
        for (int s = 0; s < sList.length; s = s + 1) {
          numElements = numElements * sList[s];
        }

        Pointer<Float> tempGradPtr = calloc<Float>(numElements);
        CudaEngine.retrieve('${gpuInputs[i].id}_grad', tempGradPtr);

        // INSTANT NATIVE ADDITION (Bypasses Dart GC and Loop entirely)
        CudaEngine.addPointers(cpuInputs[i].gradPtr, tempGradPtr, numElements);

        calloc.free(tempGradPtr);
      }
    },
    opName: opName,
    extraParams: {
      'gpu_output': gpuOutput,
      'gpu_inputs': gpuInputs,
    },
  );

  return out;
}