import 'dart:math';
import 'dart:math' as math;
import '../gpu_version/ffi/allocation_buffer.dart';
import 'tensor_gpu.dart';
import 'type_aliases.dart';

import '../gpu_version/ffi/op_codes.dart';
import '../gpu_version/ffi/command_buffer.dart';

/// /////////////////////////////////
/// Data management (0-99)         ///
/// /////////////////////////////////

GPUTensor<Matrix> reshapeVectorToMatrixGPU(GPUTensor<Vector> v, int numRows,
    int numCols, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'reshapeGPU');

  tape.putInt(OP_COPY);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'reshapeGPU',
    extraParams: {'numRows': numRows, 'numCols': numCols},
    cost: 0,
  );

  return out;
}

GPUTensor<Tensor3D> reshapeMatrixTo3DGPU(GPUTensor<Matrix> m, int c, int h,
    int w, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[c, h, w], allocTape,
      sourceOp: 'reshapeMatrixTo3DGPU');

  tape.putInt(OP_COPY);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'reshapeMatrixTo3DGPU',
    cost: 0,
  );

  return out;
}

GPUTensor<Matrix> reshape3DToMatrixGPU(GPUTensor<Tensor3D> t, int rows,
    int cols, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[rows, cols], allocTape,
      sourceOp: 'reshape3DToMatrixGPU');

  tape.putInt(OP_COPY);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(t.gradId);
    },
    opName: 'reshape3DToMatrixGPU',
    cost: 0,
  );

  return out;
}

GPUTensor<Matrix> flatten3DToMatrixGPU(
    GPUTensor<Tensor3D> t, CommandBuffer tape, AllocBuffer allocTape) {
  int c = t.shape[0];
  int h = t.shape[1];
  int w = t.shape[2];
  int flatSize = c * h * w;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[1, flatSize], allocTape,
      sourceOp: 'flatten3DToMatrixGPU');

  tape.putInt(OP_COPY);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(t.gradId);
    },
    opName: 'flatten3DToMatrixGPU',
    cost: 0,
  );

  return out;
}

GPUTensor<Vector> loadSampleGPU(GPUTensor<Matrix> dataset, int sampleIndex,
    CommandBuffer tape, AllocBuffer allocTape) {
  int cols = dataset.shape[1];

  List<int> outShape = <int>[cols];
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(outShape, allocTape, sourceOp: 'loadSampleGPU');

  tape.putInt(OP_LOAD_SAMPLE);
  tape.putString(dataset.id);
  tape.putString(out.id);
  tape.putPatchableInt('sample_index', sampleIndex);

  out.creator = GPUNode(
    <GPUTensor>[dataset],
        (CommandBuffer bTape) {
      bTape.putInt(OP_STORE_SAMPLE);
      bTape.putString(out.gradId);
      bTape.putString(dataset.gradId);
      bTape.putPatchableInt('sample_index', sampleIndex);
    },
    opName: 'loadSampleGPU',
  );

  return out;
}

// /////////////////////////////////
// Basic Math    (100-199)       ///
// /////////////////////////////////

/// Adds the command for addition of two GPUTensors [a] and [b] of type [T] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> addGPU<T>(
    GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'addGPU');

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'addGPU',
  );
  return out;
}

/// Adds the command for addition of two vectors [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Vector> addVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(a.shape, allocTape, sourceOp: 'addVectorGPU');

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'addVectorGPU',
  );

  return out;
}

/// Adds the command for addition of two matrices [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> addMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'add_matrixGPU');

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'add_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Adds the command for addition of two Tensor3Ds [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> add3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'add_3dGPU');

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'add_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for subtraction of two GPUTensors [a] and [b] of type [T] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> subtractGPU<T>(
    GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'subtractGPU');

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'subtractGPU',
  );

  return out;
}

GPUTensor<Vector> subtractVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int length = a.shape[0];
  List<int> shape = <int>[length];
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(shape, allocTape, sourceOp: 'subtractVectorGPU');

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'subtractVectorGPU',
    cost: length,
  );

  return out;
}

/// Appends the commands for subtraction of two matrices [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> subtractMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];
  List<int> shape = <int>[numRows, numCols];
  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(shape, allocTape, sourceOp: 'subtractMatrixGPU');

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'subtractMatrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends the commands for subtraction of two Tensor3Ds [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> subtract3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];
  List<int> shape = <int>[depth, height, width];
  GPUTensor<Tensor3D> out =
  GPUTensor<Tensor3D>.empty(shape, allocTape, sourceOp: 'subtract3DGPU');

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'subtract3DGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for element wise multiplication of two GPUTensors [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> multiplyGPU<T>(
    GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'multiplyGPU');

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(b.gradId);
    },
    opName: 'multiplyGPU',
    cost: 1,
  );

  return out;
}

GPUTensor<Scalar> multiplyScalarGPU(GPUTensor<Scalar> a, GPUTensor<Scalar> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Scalar> out =
  GPUTensor<Scalar>.empty(<int>[], allocTape, sourceOp: 'multiplyScalarGPU');

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(b.gradId);
    },
    opName: 'multiplyScalarGPU',
    cost: 1,
  );

  return out;
}

/// Appends the commands for element wise multiplication of two vectors [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Vector> elementWiseMultiplyGPU(GPUTensor<Vector> a,
    GPUTensor<Vector> b, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(a.shape, allocTape,
      sourceOp: 'elementWiseMultiplyGPU');

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(b.gradId);
    },
    opName: 'elementWiseMultiplyGPU',
    cost: 1,
  );

  return out;
}

/// Appends the commands for element wise multiplication of two Tensor3Ds [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> elementWiseMultiply3DGPU(GPUTensor<Tensor3D> a,
    GPUTensor<Tensor3D> b, CommandBuffer tape, AllocBuffer allocTape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'multiply_3dGPU');

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(b.gradId);
    },
    opName: 'multiply_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for element wise multiplication of two matrices [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> elementWiseMultiplyMatrixGPU(GPUTensor<Matrix> a,
    GPUTensor<Matrix> b, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'multiply_matrixGPU');

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(b.gradId);
    },
    opName: 'multiply_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends the commands for element wise division of two GPUTensors [a] through [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> divideGPU<T>(
    GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'divideGPU');

  tape.putInt(OP_DIVIDE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DIVIDE_BACKWARD);
      bTape.putString(a.id);
      bTape.putString(b.id);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'divideGPU',
  );

  return out;
}

GPUTensor<Vector> divideVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int length = a.shape[0];
  List<int> shape = <int>[length];
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(shape, allocTape, sourceOp: 'divideVectorGPU');

  tape.putInt(OP_DIVIDE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DIVIDE_BACKWARD);
      bTape.putString(a.id);
      bTape.putString(b.id);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'divideVectorGPU',
    cost: length,
  );

  return out;
}

/// Appends the commands for element wise division of two matrices [a] through [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> divideMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];
  List<int> shape = <int>[numRows, numCols];
  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(shape, allocTape, sourceOp: 'divideMatrixGPU');

  tape.putInt(OP_DIVIDE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DIVIDE_BACKWARD);
      bTape.putString(a.id);
      bTape.putString(b.id);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'divideMatrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends the commands for element wise division of two Tensor3Ds [a] through [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> divide3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];
  List<int> shape = <int>[depth, height, width];
  GPUTensor<Tensor3D> out =
  GPUTensor<Tensor3D>.empty(shape, allocTape, sourceOp: 'divide3DGPU');

  tape.putInt(OP_DIVIDE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DIVIDE_BACKWARD);
      bTape.putString(a.id);
      bTape.putString(b.id);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'divide3DGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for applying the e^v exponential function on vector [v] to [tape].
GPUTensor<Vector> vectorExpGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  int N = v.shape[0];
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(v.shape, allocTape, sourceOp: 'exp_vectorGPU');

  tape.putInt(OP_EXP_ELEMENTWISE);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EXP_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(out.id);
      bTape.putString(v.gradId);
    },
    opName: 'exp_vectorGPU',
    cost: N,
  );

  return out;
}

/// Appends the commands for applying the abs(v) exponential function on every element of [a] to [tape].
GPUTensor<T> absGPU<T>(
    GPUTensor<T> a, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'absGPU');

  tape.putInt(OP_ABS_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ABS_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(a.gradId);
    },
    opName: 'absGPU',
  );

  return out;
}

/// Appends the commands for applying the sqrt(v) exponential function on every element of [v] to [tape].
GPUTensor<T> sqrtGPU<T>(
    GPUTensor<T> a, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'sqrtGPU');

  tape.putInt(OP_SQRT_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SQRT_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(out.id);
      bTape.putString(a.gradId);
    },
    opName: 'sqrtGPU',
  );

  return out;
}

/// Appends the commands for applying the log(v) exponential function on every element of [v] to [tape].
GPUTensor<T> logGPU<T>(
    GPUTensor<T> a, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'logGPU');

  tape.putInt(OP_LOG_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_LOG_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(a.gradId);
    },
    opName: 'logGPU',
  );

  return out;
}

/// Appends the commands for applying pow([v],[exponent]) exponential function on every element of [v] to [tape].
GPUTensor<T> powGPU<T>(
    GPUTensor<T> a, double exponent, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'powGPU');

  tape.putInt(OP_POW_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);
  tape.putFloat(exponent);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_POW_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(a.gradId);
      bTape.putFloat(exponent);
    },
    opName: 'powGPU',
  );

  return out;
}

/// Appends the commands for clamping every element of [v] between [minVal] - [maxVal] to [tape].
GPUTensor<T> clampGPU<T>(GPUTensor<T> a, double minVal, double maxVal,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(a.shape, allocTape, sourceOp: 'clampGPU');

  tape.putInt(OP_CLAMP_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);
  tape.putFloat(minVal);
  tape.putFloat(maxVal);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CLAMP_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.id);
      bTape.putString(a.gradId);
      bTape.putFloat(minVal);
      bTape.putFloat(maxVal);
    },
    opName: 'clampGPU',
  );

  return out;
}

/// /////////////////////////////////
/// Matrix Operations  (200-299)  ///
/// /////////////////////////////////

/// Appends the commands for matrix multiplication of [a] with [b] to [tape].
GPUTensor<Matrix> matMulGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int M = a.shape[0];
  int N = a.shape[1];
  int P = b.shape[1];

  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(<int>[M, P], allocTape, sourceOp: 'matMulGPU');

  tape.putInt(OP_MATMUL);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putBool(false);
  tape.putBool(false);
  tape.putFloat(1.0);
  tape.putFloat(0.0);
  tape.putBool(true);

  int cost = 2 * M * N * P;

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MATMUL);
      bTape.putString(out.gradId);
      bTape.putString(b.id);
      bTape.putString(a.gradId);
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true);

      bTape.putInt(OP_MATMUL);
      bTape.putString(a.id);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
      bTape.putBool(true);
      bTape.putBool(false);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true);
    },
    opName: 'matMulGPU',
    cost: cost,
  );

  return out;
}

/// Appends the commands for multiplication of matrix[a] with vector[b] to [tape].
GPUTensor<Vector> matVecMulGPU(GPUTensor<Matrix> mMat, GPUTensor<Vector> v,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = mMat.shape[0];
  int numCols = mMat.shape[1];

  GPUTensor<Vector> out = GPUTensor<Vector>.empty(<int>[numRows], allocTape,
      sourceOp: 'matVecMulGPU');

  tape.putInt(OP_MATMUL);
  tape.putString(mMat.id);
  tape.putString(v.id);
  tape.putString(out.id);
  tape.putBool(false);
  tape.putBool(false);
  tape.putFloat(1.0);
  tape.putFloat(0.0);
  tape.putBool(false);

  out.creator = GPUNode(
    <GPUTensor>[mMat, v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MATMUL);
      bTape.putString(out.gradId);
      bTape.putString(v.id);
      bTape.putString(mMat.gradId);
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(false);

      bTape.putInt(OP_MATMUL);
      bTape.putString(mMat.id);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
      bTape.putBool(true);
      bTape.putBool(false);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(false);
    },
    opName: 'matVecMulGPU',
    cost: 2 * numRows * numCols,
  );

  return out;
}

/// Appends a command for switching rows and columns to [tape].
GPUTensor<Matrix> transposeGPU(
    GPUTensor<Matrix> a, CommandBuffer tape, AllocBuffer allocTape) {
  int M = a.shape[0];
  int N = a.shape[1];

  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(<int>[N, M], allocTape, sourceOp: 'transposeGPU');
  GPUTensor<Matrix> tmpGrad = GPUTensor<Matrix>.empty(<int>[M, N], allocTape,
      requiresGrad: false, sourceOp: 'transposeGPU:tmpGrad');
  out.subMap["tmpGrad"] = tmpGrad;

  tape.putInt(OP_TRANSPOSE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TRANSPOSE);
      bTape.putString(out.gradId);
      bTape.putString(tmpGrad.id);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(tmpGrad.id);
      bTape.putString(a.gradId);
    },
    opName: 'transposeGPU',
    extraParams: {'tmpGrad': tmpGrad},
    cost: 0,
  );

  return out;
}

/// Appends a command for broadcast add a vector [v] to [tape].
GPUTensor<Matrix> addMatrixAndVectorGPU(GPUTensor<Matrix> m,
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'addMatrixAndVectorGPU');

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);

      bTape.putInt(OP_SUM_REDUCE_COLUMNS);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'addMatrixAndVectorGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends a command for adding a scalar [b] to every element of a matrix [m] to [tape].
GPUTensor<Matrix> addScalarMatrixGPU(GPUTensor<Matrix> m,
    GPUTensor<Scalar> b, CommandBuffer tape, AllocBuffer allocTape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  List<int> shape = <int>[rows, cols];
  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(shape, allocTape, sourceOp: 'addScalarToMatrixGPU');

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);

      bTape.putInt(OP_SUM_REDUCE);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'addScalarToMatrixGPU',
    cost: rows * cols,
  );

  return out;
}

/// Appends a command for adding a scalar [b] to every element of a vector [v] to [tape].
GPUTensor<Vector> addScalarVectorGPU(GPUTensor<Vector> v, double scalar,
    CommandBuffer tape, AllocBuffer allocTape) {
  int length = v.shape[0];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(<int>[length], allocTape,
      sourceOp: 'addScalarVectorGPU');

  tape.putInt(OP_ADD_SCALAR);
  tape.putString(v.id);
  tape.putString(out.id);
  tape.putFloat(scalar);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'addScalarVectorGPU',
    cost: length,
  );

  return out;
}

/// Appends a command for adding a scalar [scalar] to every element of a Tensor3D [v] to [tape].
GPUTensor<Tensor3D> addScalar3DGPU(GPUTensor<Tensor3D> t, double scalar,
    CommandBuffer tape, AllocBuffer allocTape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'addScalar3DGPU');

  tape.putInt(OP_ADD_SCALAR);
  tape.putString(t.id);
  tape.putString(out.id);
  tape.putFloat(scalar);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(t.gradId);
    },
    opName: 'addScalar3DGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for broadcast additions of a matrix wrapped scalar [b] to a full matrix [m].
GPUTensor<Matrix> addBiasToFeatureMapGPU(GPUTensor<Matrix> m,
    GPUTensor<Matrix> b, CommandBuffer tape, AllocBuffer allocTape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  List<int> shape = <int>[rows, cols];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(shape, allocTape,
      sourceOp: 'addBiasToFeatureMapGPU');

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);

      bTape.putInt(OP_SUM_REDUCE);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'addBiasToFeatureMapGPU',
    cost: rows * cols,
  );

  return out;
}

/// Appends commands for addition of a 1D bias vector [b] to a 2D matrix [m] via broadcasting over the matrix to the [tape].
GPUTensor<Matrix> addBiasToMatMulOutGPU(GPUTensor<Matrix> m,
    GPUTensor<Vector> b, CommandBuffer tape, AllocBuffer allocTape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[rows, cols], allocTape,
      sourceOp: 'addBiasToMatMulOutGPU');

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);

      bTape.putInt(OP_SUM_REDUCE_ROWS);
      bTape.putString(out.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'addBiasToMatMulOutGPU',
    cost: rows * cols,
  );

  return out;
}

/// Appends commands for addition of a 1D bias vector [b] to a 2D matrix [m] via broadcasting over the matrix to the [tape].
/// This command is inference only! No gradient path is calculated.
GPUTensor<Matrix> broadcastAddVectorToMatrixGPU(GPUTensor<Matrix> m,
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(m.shape, allocTape,
      sourceOp: 'broadcastAddVectorToMatrixGPU');

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, v],
        (CommandBuffer bTape) {},
    opName: 'broadcastAddVectorToMatrixGPU',
    cost: m.shape[0] * m.shape[1],
  );

  return out;
}

/// Appends commands for additions of a scalar constant [s] to every element of matrix [m] to [tape].
GPUTensor<Matrix> scaleMatrixGPU(GPUTensor<Matrix> m, double s,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'scaleMatrixGPU');

  tape.putInt(OP_SCALE_MATRIX);
  tape.putString(m.id);
  tape.putString(out.id);
  tape.putFloat(s);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SCALE_MATRIX_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
      bTape.putFloat(s);
    },
    opName: 'scaleMatrixGPU',
    cost: numRows * numCols,
    extraParams: {'s': s},
  );

  return out;
}

/// /////////////////////////////////
/// Activations       (300-399)   ///
/// /////////////////////////////////

/// Appends commands for applying Relu on every element of vector [v] to [tape].
GPUTensor<Vector> reluGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(v.shape, allocTape, sourceOp: 'reluGPU');

  tape.putInt(OP_RELU);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_RELU_BACKWARD);
      bTape.putString(v.id);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'reluGPU',
    cost: v.shape[0],
  );

  return out;
}

/// Appends commands for applying Relu on every element of matrix [m] to [tape].
GPUTensor<Matrix> reluMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'relu_matrixGPU');

  tape.putInt(OP_RELU);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_RELU_BACKWARD);
      bTape.putString(m.id);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'relu_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the sigmoid function on every element of [m] (type [t]) to [tape].
GPUTensor<T> sigmoidScalarGPU<T>(
    GPUTensor<T> s, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<T> out =
  GPUTensor<T>.empty(s.shape, allocTape, sourceOp: 'sigmoidScalarGPU');

  tape.putInt(OP_SIGMOID);
  tape.putString(s.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[s],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(s.gradId);
    },
    opName: 'sigmoidScalarGPU',
    cost: 1,
  );

  return out;
}

/// Appends commands for applying the sigmoid function on every element of vector [v] to [tape].
GPUTensor<Vector> sigmoidGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(v.shape, allocTape, sourceOp: 'sigmoidGPU');

  tape.putInt(OP_SIGMOID);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'sigmoidGPU',
    cost: v.shape[0],
  );

  return out;
}

/// Appends commands for applying the sigmoid function on every element of matrix [m] to [tape].
GPUTensor<Matrix> sigmoidMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'sigmoid_matrixGPU');

  tape.putInt(OP_SIGMOID);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'sigmoid_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the sigmoid function on every element of Tensor3D [t] to [tape].
GPUTensor<Tensor3D> sigmoid3DGPU(
    GPUTensor<Tensor3D> t, CommandBuffer tape, AllocBuffer allocTape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'sigmoid_3dGPU');

  tape.putInt(OP_SIGMOID);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(t.gradId);
    },
    opName: 'sigmoid_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for applying the tanh function on every element of vector [t] to [tape].
GPUTensor<Vector> vectorTanhGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  int N = v.shape[0];
  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(v.shape, allocTape, sourceOp: 'tanh_vectorGPU');

  tape.putInt(OP_TANH);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'tanh_vectorGPU',
    cost: N,
  );

  return out;
}

/// Appends commands for applying the tanh function on every element of matrix [m] to [tape].
GPUTensor<Matrix> tanhMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'tanh_matrixGPU');

  tape.putInt(OP_TANH);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'tanh_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the tanh function on every element of Tensor3D [t] to [tape].
GPUTensor<Tensor3D> tanh3DGPU(
    GPUTensor<Tensor3D> t, CommandBuffer tape, AllocBuffer allocTape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'tanh_3dGPU');

  tape.putInt(OP_TANH);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.gradId);
      bTape.putString(t.gradId);
    },
    opName: 'tanh_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for applying the gelu activation function on every element of vector [v] to [tape].
GPUTensor<Vector> geluGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  int n = v.shape[0];

  GPUTensor<Vector> out =
  GPUTensor<Vector>.empty(v.shape, allocTape, sourceOp: 'geluGPU');

  tape.putInt(OP_GELU_FORWARD);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_GELU_BACKWARD);
      bTape.putString(v.id);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'geluGPU',
    cost: n,
  );

  return out;
}

/// Appends commands for applying the gelu activation function on every element of matrix [m] to [tape].
GPUTensor<Matrix> geluMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'gelu_matrixGPU');

  tape.putInt(OP_GELU_FORWARD);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_GELU_BACKWARD);
      bTape.putString(m.id);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'gelu_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the softmax function on every element of matrix [m] to [tape].
GPUTensor<Matrix> softmaxMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'softmax_matrixGPU');

  tape.putInt(OP_SOFTMAX_FORWARD);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SOFTMAX_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(out.id);
      bTape.putString(m.gradId);
    },
    opName: 'softmax_matrixGPU',
    cost: numRows * numCols * 3,
  );

  return out;
}

/// /////////////////////////////////
/// Loss Functions    (400-499)   ///
/// /////////////////////////////////

/// Appends commands for calculating the binary cross entropy over the elements of [m] of type [T] to [tape].
GPUTensor<Scalar> binaryCrossEntropyGPU<T>(GPUTensor<T> prediction,
    GPUTensor<T> target, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Scalar> out = GPUTensor<Scalar>.empty(<int>[], allocTape,
      sourceOp: 'binaryCrossEntropyGPU');

  tape.putInt(OP_BCE_LOSS_FORWARD);
  tape.putString(prediction.id);
  tape.putString(target.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[prediction, target],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BCE_LOSS_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(prediction.id);
      bTape.putString(target.id);
      bTape.putString(prediction.gradId);
    },
    opName: 'binaryCrossEntropyGPU',
    cost: 1,
  );

  return out;
}

/// Appends commands for calculating the mean square error over between the vectors [predictions] and [targets] to [tape].
GPUTensor<Scalar> mseGPU(GPUTensor<Vector> predictions,
    GPUTensor<Vector> targets, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Scalar> out =
  GPUTensor<Scalar>.empty(<int>[], allocTape, sourceOp: 'mseGPU');

  tape.putInt(OP_MSE_LOSS_FORWARD);
  tape.putString(predictions.id);
  tape.putString(targets.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[predictions, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(predictions.id);
      bTape.putString(targets.id);
      bTape.putString(predictions.gradId);
    },
    opName: 'mseGPU',
    cost: 3 * predictions.shape[0],
  );

  return out;
}

/// Appends commands for calculating the mean square error over between the matrices [predictions] and [targets] to [tape].
GPUTensor<Scalar> mseMatrixGPU(GPUTensor<Matrix> predictions,
    GPUTensor<Matrix> targets, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = predictions.shape[0];
  int numCols = predictions.shape[1];

  GPUTensor<Scalar> out =
  GPUTensor<Scalar>.empty(<int>[], allocTape, sourceOp: 'mse_matrixGPU');

  tape.putInt(OP_MSE_LOSS_FORWARD);
  tape.putString(predictions.id);
  tape.putString(targets.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[predictions, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(predictions.id);
      bTape.putString(targets.id);
      bTape.putString(predictions.gradId);
    },
    opName: 'mse_matrixGPU',
    cost: 3 * numRows * numCols,
  );

  return out;
}

void sgdUpdateGPU(GPUTensor<dynamic> data, double lr, CommandBuffer tape) {
  tape.putInt(OP_SGD_UPDATE);
  tape.putString(data.id);
  tape.putString(data.gradId);
  tape.putPatchableFloat('lr', lr);
}

void adamUpdateGPU(
    GPUTensor<dynamic> data,
    GPUTensor<dynamic> m,
    GPUTensor<dynamic> v,
    double lr,
    double beta1,
    double beta2,
    double eps,
    int step,
    double weightDecay,
    CommandBuffer tape) {
  tape.putInt(OP_ADAM_UPDATE);
  tape.putString(data.id);
  tape.putString(data.gradId);
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putPatchableFloat('lr', lr);
  tape.putFloat(beta1);
  tape.putFloat(beta2);
  tape.putFloat(eps);
  tape.putPatchableInt('step', step);
  tape.putFloat(weightDecay);
}

void clipGradValueGPU(
    GPUTensor<dynamic> tensor, double clipValue, CommandBuffer tape) {
  tape.putInt(OP_CLIP_GRAD_VALUE);
  tape.putString(tensor.gradId);
  tape.putPatchableFloat('clip_value', clipValue);
}


/// /////////////////////////////////
/// Reductions    (600-699)       ///
/// /////////////////////////////////

/// Appends commands for summing all elements of vector [v] to [tape].
GPUTensor<Scalar> sumGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  int N = v.shape[0];
  GPUTensor<Scalar> out =
  GPUTensor<Scalar>.empty(<int>[], allocTape, sourceOp: 'sum_vectorGPU');

  tape.putInt(OP_SUM_REDUCE);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SUM_REDUCE_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(v.gradId);
    },
    opName: 'sum_vectorGPU',
    cost: N,
  );

  return out;
}

/// Appends commands for summing all elements of matrix [m] to [tape].
GPUTensor<Scalar> sumMatrixGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Scalar> out =
  GPUTensor<Scalar>.empty(<int>[], allocTape, sourceOp: 'sum_matrixGPU');

  tape.putInt(OP_SUM_REDUCE);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SUM_REDUCE_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'sum_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Acts as a high performance lookup of rows in a given matrix [weights] via their row indices [indices].
/// Can be used for fast odering of elements.
/// The operation is appended to [tape] for execution.
GPUTensor<Matrix> embeddingLookupGPU(GPUTensor<Vector> indices,
    GPUTensor<Matrix> weights, CommandBuffer tape, AllocBuffer allocTape) {
  int numIndices = indices.shape[0];
  int embeddingDim = weights.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numIndices, embeddingDim], allocTape,
      sourceOp: 'embedding_lookup_gpu');

  tape.putInt(OP_EMBEDDING_FORWARD);
  tape.putString(indices.id);
  tape.putString(weights.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[weights],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EMBEDDING_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(indices.id);
      bTape.putString(weights.gradId);
    },
    opName: 'embedding_lookup_gpu',
    cost: numIndices * embeddingDim,
  );

  return out;
}

/// Acts as a high performance lookup of rows in a given matrix [weights] via their row indices [indices].
/// In contrast to [embeddingLookupGPU] this method performs a batch process to speed up lookup.
/// Can be used for fast odering of elements.
/// The operation is appended to [tape] for execution.
GPUTensor<Tensor3D> embeddingLookupBatchGPU(GPUTensor<Matrix> batchIndices,
    GPUTensor<Matrix> weights, CommandBuffer tape, AllocBuffer allocTape) {
  int batchSize = batchIndices.shape[0];
  int sequenceLength = batchIndices.shape[1];
  int embeddingDim = weights.shape[1];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[batchSize, sequenceLength, embeddingDim], allocTape,
      sourceOp: 'embedding_lookup_batch_gpu');

  tape.putInt(OP_EMBEDDING_FORWARD);
  tape.putString(batchIndices.id);
  tape.putString(weights.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[weights],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EMBEDDING_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(batchIndices.id);
      bTape.putString(weights.gradId);
    },
    opName: 'embedding_lookup_batch_gpu',
    cost: batchSize * sequenceLength * embeddingDim,
  );

  return out;
}

/// Sums all matrix elements of [m] via their column order returning the sum of each column as a vector.
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> sumReduceColumnsGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int cols = m.shape[1];

  List<int> outShape = <int>[cols];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape, allocTape,
      sourceOp: 'sumReduceColumnsGPU');

  tape.putInt(OP_SUM_REDUCE_COLUMNS);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BROADCAST_ADD);
      bTape.putString(m.gradId);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'sumReduceColumnsGPU',
  );

  return out;
}

/// Sums all matrix elements of [m] via their row order returning the sum of each row as a vector.
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> sumReduceRowsGPU(
    GPUTensor<Matrix> m, CommandBuffer tape, AllocBuffer allocTape) {
  int rows = m.shape[0];

  List<int> outShape = <int>[rows];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape, allocTape,
      sourceOp: 'sumReduceRowsGPU');

  tape.putInt(OP_SUM_REDUCE_ROWS);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BROADCAST_ADD);
      bTape.putString(m.gradId);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
    },
    opName: 'sumReduceRowsGPU',
  );

  return out;
}

/// /////////////////////////////////
/// Tensor Manipulations(700-799) ///
/// /////////////////////////////////

/// Extracts a continous subset of columns out of a matrix [m] given by range [startCol]-[endCol].
/// The operation is appended to [tape] for execution.
GPUTensor<Matrix> sliceColumnGPU(GPUTensor<Matrix> input, int startCol,
    int endCol, CommandBuffer tape, AllocBuffer allocTape) {
  int rows = input.shape[0];
  int outCols = endCol - startCol;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[rows, outCols], allocTape,
      sourceOp: 'slice_column');

  tape.putInt(OP_SLICE_COLUMN);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(startCol);
  tape.putInt(endCol);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SLICE_COLUMN_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.gradId);
      bTape.putInt(startCol);
      bTape.putInt(endCol);
    },
    opName: 'slice_column',
    cost: rows * outCols,
  );

  return out;
}

/// Extracts a single row out of a matrix [m] given by [rowIndex].
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> selectRowGPU(GPUTensor<Matrix> m, int rowIndex,
    CommandBuffer tape, AllocBuffer allocTape) {
  int numCols = m.shape[1];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(<int>[numCols], allocTape,
      sourceOp: 'selectRowGPU');

  tape.putInt(OP_SLICE_ROW);
  tape.putString(m.id);
  tape.putString(out.id);
  tape.putInt(rowIndex);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SLICE_ROW_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(m.gradId);
      bTape.putInt(rowIndex);
    },
    opName: 'selectRowGPU',
    extraParams: {'rowIndex': rowIndex},
    cost: 0,
  );

  return out;
}

/// Extracts a single matrix out of a Tensor3D [t] given by [index].
/// The operation is appended to [tape] for execution.
GPUTensor<Matrix> selectMatrixFrom3DGPU(GPUTensor<Tensor3D> t, int index,
    CommandBuffer tape, AllocBuffer allocTape) {
  int height = t.shape[1];
  int width = t.shape[2];

  List<int> outShape = <int>[height, width];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(outShape, allocTape,
      sourceOp: 'selectMatrixFrom3DGPU');

  tape.putInt(OP_SLICE_ROW);
  tape.putString(t.id);
  tape.putString(out.id);
  tape.putInt(index);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_SLICE_ROW_BACKWARD);
      backwardTape.putString(out.gradId);
      backwardTape.putString(t.gradId);
      backwardTape.putInt(index);
    },
    opName: 'selectMatrixFrom3DGPU',
  );

  return out;
}

/// Appends two vectors [a] and [b] together in order [a-b].
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> concatenateGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  int aLength = a.shape[0];
  int totalLength = aLength + b.shape[0];

  GPUTensor<Vector> out = GPUTensor<Vector>.empty(
      <int>[totalLength], allocTape,
      sourceOp: 'concat_vectorGPU');

  tape.putInt(OP_CONCATENATE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putInt(0);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONCATENATE_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
      bTape.putInt(0);
      bTape.putInt(aLength);
    },
    opName: 'concat_vectorGPU',
    cost: 0,
  );

  return out;
}

/// Appends a list of matrices [matrices] together in column order.
/// The operation is appended to [tape] for execution.
GPUTensor<Matrix> concatenateMatricesByColumnGPU(
    List<GPUTensor<Matrix>> matrices,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  GPUTensor<Matrix> result = matrices[0];
  List<GPUTensor<Matrix>> intermediates = <GPUTensor<Matrix>>[];

  for (int i = 1; i < matrices.length; i = i + 1) {
    GPUTensor<Matrix> a = result;
    GPUTensor<Matrix> b = matrices[i];

    int rows = a.shape[0];
    int colsA = a.shape[1];
    int colsB = b.shape[1];
    int totalCols = colsA + colsB;

    bool isLast = i == matrices.length - 1;
    GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
        <int>[rows, totalCols], allocTape,
        sourceOp: 'concat_matrix_colGPU');
    if (isLast == false) {
      intermediates.add(out);
    }

    tape.putInt(OP_CONCATENATE);
    tape.putString(a.id);
    tape.putString(b.id);
    tape.putString(out.id);
    tape.putInt(1);

    out.creator = GPUNode(
      <GPUTensor>[a, b],
          (CommandBuffer bTape) {
        bTape.putInt(OP_CONCATENATE_BACKWARD);
        bTape.putString(out.gradId);
        bTape.putString(a.gradId);
        bTape.putString(b.gradId);
        bTape.putInt(1);
        bTape.putInt(colsA);
      },
      opName: 'concat_matrix_colGPU',
      cost: 0,
    );

    result = out;
  }

  for (int i = 0; i < intermediates.length; i = i + 1) {
    result.subMap['intermediate_$i'] = intermediates[i];
  }

  return result;
}

/// Appends two Tensor3Ds [a] and [b] together in order [a-b].
/// The operation is appended to [tape] for execution.
GPUTensor<Tensor3D> concatenate3DGPU(GPUTensor<Tensor3D> a,
    GPUTensor<Tensor3D> b, CommandBuffer tape, AllocBuffer allocTape) {
  int aDepth = a.shape[0];
  int bDepth = b.shape[0];
  int totalDepth = aDepth + bDepth;
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[totalDepth, height, width], allocTape,
      sourceOp: 'concat_3dGPU');

  tape.putInt(OP_CONCATENATE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putInt(0);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONCATENATE_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(a.gradId);
      bTape.putString(b.gradId);
      bTape.putInt(0);
      bTape.putInt(aDepth);
    },
    opName: 'concat_3dGPU',
    cost: 0,
  );

  return out;
}

/// Appends a list of matrices [matrices] together together to form a Tensor3D.
/// The operation is appended to [tape] for execution.
GPUTensor<Tensor3D> stackMatricesGPU(List<GPUTensor<Matrix>> matrices,
    CommandBuffer tape, AllocBuffer allocTape) {
  int count = matrices.length;
  int rows = matrices[0].shape[0];
  int cols = matrices[0].shape[1];

  List<int> shape = <int>[count, rows, cols];
  GPUTensor<Tensor3D> out =
  GPUTensor<Tensor3D>.empty(shape, allocTape, sourceOp: 'stackMatricesGPU');

  tape.putInt(OP_STACK_ROWS);
  tape.putInt(count);
  for (int i = 0; i < count; i = i + 1) {
    tape.putString(matrices[i].id);
  }
  tape.putString(out.id);
  tape.putInt(0);

  out.creator = GPUNode(
    <GPUTensor>[...matrices],
        (CommandBuffer bTape) {
      bTape.putInt(OP_STACK_ROWS_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putInt(count);
      for (int i = 0; i < count; i = i + 1) {
        bTape.putString(matrices[i].gradId);
      }
      bTape.putInt(0);
    },
    opName: 'stackMatricesGPU',
    cost: count * rows * cols,
  );

  return out;
}

/// Constructs a matrix from smaller matrices packing them into a bigger one in order. Primarily used for attention-heads.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> scatterHeadsGPU(List<GPUTensor<Matrix>> heads, int dModel,
    CommandBuffer tape, AllocBuffer allocTape) {
  int seqLen = heads[0].shape[0];
  int dHead = heads[0].shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[seqLen, dModel], allocTape,
      sourceOp: 'scatter_heads');

  tape.putInt(OP_FILL);
  tape.putString(out.id);
  tape.putFloat(0.0);

  for (int i = 0; i < heads.length; i = i + 1) {
    int startCol = i * dHead;
    int endCol = startCol + dHead;

    tape.putInt(OP_SLICE_COLUMN_BACKWARD);
    tape.putString(heads[i].id);
    tape.putString(out.id);
    tape.putInt(startCol);
    tape.putInt(endCol);
  }

  out.creator = GPUNode(
    <GPUTensor>[...heads],
        (CommandBuffer bTape) {
      for (int i = 0; i < heads.length; i = i + 1) {
        int startCol = i * dHead;
        int endCol = startCol + dHead;

        bTape.putInt(OP_SLICE_COLUMN);
        bTape.putString(out.gradId);
        bTape.putString(heads[i].gradId);
        bTape.putInt(startCol);
        bTape.putInt(endCol);
      }
    },
    opName: 'scatter_heads',
    cost: seqLen * dModel,
  );

  return out;
}

/// Expands a matrix [input] symmetrically to all sides with 0.0 padding with length of [padSize].
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> padMatrixGPU(GPUTensor<Matrix> input, int padSize,
    CommandBuffer tape, AllocBuffer allocTape) {
  int inHeight = input.shape[0];
  int inWidth = input.shape[1];
  int outHeight = inHeight + 2 * padSize;
  int outWidth = inWidth + 2 * padSize;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[outHeight, outWidth], allocTape,
      sourceOp: 'padMatrixGPU');

  tape.putInt(OP_PAD2D);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(padSize);
  tape.putInt(padSize);
  tape.putInt(padSize);
  tape.putInt(padSize);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_PAD2D_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.gradId);
      bTape.putInt(padSize);
      bTape.putInt(padSize);
      bTape.putInt(padSize);
      bTape.putInt(padSize);
    },
    opName: 'padMatrixGPU',
    cost: outHeight * outWidth,
  );

  return out;
}

/// /////////////////////////////////
/// Advanced Layers (800- 999)    ///
/// /////////////////////////////////

/// Performs an advanced 2D convolution on multiple channels with [input] and [weight], alongside a [bias] vector.
///
/// Supports configurable kernel dimensions ([kH], [kW]), [padding] ('valid' or 'same'), and strides ([strideH], [strideW]).
/// Records the forward pass to the [tape] and registers distinct backward passes for the input, weight, and bias gradients.
GPUTensor<Tensor3D> conv2dMultiChannelGPU(
    GPUTensor<dynamic> input,
    GPUTensor<Tensor3D> weight,
    GPUTensor<Vector> bias,
    int kH,
    int kW,
    CommandBuffer tape,
    AllocBuffer allocTape,
    {String padding = 'valid',
      int strideH = 1,
      int strideW = 1}) {
  int inChannels = input.shape.length == 2 ? 1 : input.shape[0];
  int inHeight = input.shape.length == 2 ? input.shape[0] : input.shape[1];
  int inWidth = input.shape.length == 2 ? input.shape[1] : input.shape[2];

  int outChannels = weight.shape[0];

  int padT = 0;
  int padL = 0;
  int outHeight = (inHeight - kH) ~/ strideH + 1;
  int outWidth = (inWidth - kW) ~/ strideW + 1;

  if (padding == 'same') {
    padT = (kH - 1) ~/ 2;
    padL = (kW - 1) ~/ 2;
    outHeight = (inHeight + 2 * padT - kH) ~/ strideH + 1;
    outWidth = (inWidth + 2 * padL - kW) ~/ strideW + 1;
  }

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[outChannels, outHeight, outWidth], allocTape,
      sourceOp: 'conv2dMultiChannelGPU');

  tape.putInt(OP_CONV2D_MULTI_FORWARD);
  tape.putString(input.id);
  tape.putString(weight.id);
  tape.putString(bias.id);
  tape.putString(out.id);
  tape.putInt(inChannels);
  tape.putInt(outChannels);
  tape.putInt(kH);
  tape.putInt(kW);
  tape.putInt(padT);
  tape.putInt(padL);
  tape.putInt(strideH);
  tape.putInt(strideW);

  int cost = outHeight * outWidth * outChannels * inChannels * kH * kW * 2;

  out.creator = GPUNode(
    <GPUTensor>[input, weight, bias],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONV2D_MULTI_BACKWARD_INPUT);
      bTape.putString(out.gradId);
      bTape.putString(weight.id);
      bTape.putString(input.gradId);
      bTape.putInt(inChannels);
      bTape.putInt(outChannels);
      bTape.putInt(kH);
      bTape.putInt(kW);
      bTape.putInt(padT);
      bTape.putInt(padL);
      bTape.putInt(strideH);
      bTape.putInt(strideW);

      bTape.putInt(OP_CONV2D_MULTI_BACKWARD_WEIGHT);
      bTape.putString(input.id);
      bTape.putString(out.gradId);
      bTape.putString(weight.gradId);
      bTape.putString(bias.gradId);
      bTape.putInt(inChannels);
      bTape.putInt(outChannels);
      bTape.putInt(kH);
      bTape.putInt(kW);
      bTape.putInt(padT);
      bTape.putInt(padL);
      bTape.putInt(strideH);
      bTape.putInt(strideW);
    },
    opName: 'conv2dMultiChannelGPU',
    cost: cost,
  );

  return out;
}

/// Computes a basic, single-channel 2D convolution between an [input] matrix and a [kernel] matrix.
///
/// Applies strictly 'valid' padding with a stride of 1. Records the forward operation to the [tape]
/// and registers individual backward pass kernels for both the input and the kernel gradients.
GPUTensor<Matrix> conv2dSimpleGPU(GPUTensor<Matrix> input,
    GPUTensor<Matrix> kernel, CommandBuffer tape, AllocBuffer allocTape) {
  int inH = input.shape[0];
  int inW = input.shape[1];
  int kH = kernel.shape[0];
  int kW = kernel.shape[1];

  int outH = inH - kH + 1;
  int outW = inW - kW + 1;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[outH, outW], allocTape,
      sourceOp: 'conv2dSimpleGPU');

  tape.putInt(OP_CONV2D_FORWARD);
  tape.putString(input.id);
  tape.putString(kernel.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[input, kernel],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONV2D_BACKWARD_INPUT);
      bTape.putString(out.gradId);
      bTape.putString(kernel.id);
      bTape.putString(input.gradId);

      bTape.putInt(OP_CONV2D_BACKWARD_KERNEL);
      bTape.putString(input.id);
      bTape.putString(out.gradId);
      bTape.putString(kernel.gradId);
    },
    opName: 'conv2dSimpleGPU',
    cost: outH * outW * kH * kW,
  );

  return out;
}

/// Extracts sliding local pathces fromm an image Tensor and flattens them into columns of a matrix.
/// Kernel dimensions [kH] and [kW] need to be set.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> im2colGPU(GPUTensor<dynamic> input, int kH, int kW,
    CommandBuffer tape, AllocBuffer allocTape) {
  int inChannels = input.shape.length == 2 ? 1 : input.shape[0];
  int inH = input.shape.length == 2 ? input.shape[0] : input.shape[1];
  int inW = input.shape.length == 2 ? input.shape[1] : input.shape[2];

  int outH = inH - kH + 1;
  int outW = inW - kW + 1;

  int rows = inChannels * kH * kW;
  int cols = outH * outW;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[rows, cols], allocTape,
      sourceOp: 'im2colGPU');

  tape.putInt(OP_IM2COL);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(kH);
  tape.putInt(kW);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_COL2IM);
      bTape.putString(out.gradId);
      bTape.putString(input.gradId);
      bTape.putInt(kH);
      bTape.putInt(kW);
    },
    opName: 'im2colGPU',
    cost: inChannels * kH * kW * outH * outW,
  );

  return out;
}

/// Slides a pool of size [poolSize] over vector [input]. The resulting vector consists of the maximum elements in each window.
/// The value of [stride] tells how big the step size of the sliding window is.
/// The operations are appended to [tape] for execution.
GPUTensor<Vector> maxPool1dGPU(GPUTensor<Vector> input, int poolSize,
    int stride, CommandBuffer tape, AllocBuffer allocTape) {
  int inputSize = input.shape[0];
  int outputSize = (inputSize - poolSize) ~/ stride + 1;

  GPUTensor<Vector> out = GPUTensor<Vector>.empty(
      <int>[outputSize], allocTape,
      sourceOp: 'maxPool1dGPU');
  GPUTensor<Vector> indices = GPUTensor<Vector>.empty(
      <int>[outputSize], allocTape,
      requiresGrad: false, sourceOp: 'maxPool1dGPU:indices');
  out.subMap["indices"] = indices;
  tape.putInt(OP_MAX_POOL_1D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(indices.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MAX_POOL_1D_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(indices.id);
      bTape.putString(input.gradId);
    },
    opName: 'maxPool1dGPU',
    cost: inputSize,
  );

  return out;
}

/// Slides a pool of size [poolSize]x[poolSize] over matrix [input]. The resulting matrix consists of the maximum elements in each window.
/// The value of [stride] tells how big the step size of the sliding window is in x and y direction.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> maxPool2dGPU(GPUTensor<Matrix> input, int poolSize,
    int stride, CommandBuffer tape, AllocBuffer allocTape) {
  int inputHeight = input.shape[0];
  int inputWidth = input.shape[1];
  int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
  int outputWidth = (inputWidth - poolSize) ~/ stride + 1;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[outputHeight, outputWidth], allocTape,
      sourceOp: 'maxPool2dGPU');
  GPUTensor<Matrix> indices = GPUTensor<Matrix>.empty(
      <int>[outputHeight, outputWidth], allocTape,
      requiresGrad: false, sourceOp: 'maxPool2dGPU:indices');
  out.subMap["indices"] = indices;

  tape.putInt(OP_MAX_POOL_2D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(indices.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MAX_POOL_2D_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(indices.id);
      bTape.putString(input.gradId);
    },
    opName: 'maxPool2dGPU',
    cost: inputHeight * inputWidth,
  );

  return out;
}

/// Slides a pool of size [poolSize]x[poolSize] over matrix [input]. The resulting matrix consists of the average of the elements in each window.
/// The value of [stride] tells how big the step size of the sliding window is.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> avgPool2dGPU(GPUTensor<Matrix> input, int poolSize,
    int stride, CommandBuffer tape, AllocBuffer allocTape) {
  int inHeight = input.shape[0];
  int inWidth = input.shape[1];

  int outHeight = (inHeight - poolSize) ~/ stride + 1;
  int outWidth = (inWidth - poolSize) ~/ stride + 1;

  List<int> outShape = <int>[outHeight, outWidth];
  GPUTensor<Matrix> out =
  GPUTensor<Matrix>.empty(outShape, allocTape, sourceOp: 'avg_pool_2d_gpu');

  tape.putInt(OP_AVG_POOL_2D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_AVG_POOL_2D_BACKWARD);
      backwardTape.putString(out.gradId);
      backwardTape.putString(input.gradId);
      backwardTape.putInt(poolSize);
      backwardTape.putInt(stride);
    },
    opName: 'avg_pool_2d_gpu',
  );

  return out;
}

/// Collapses a matrix [input] to a vector of their average values of their feature columns.
/// The operations are appended to [tape] for execution.
GPUTensor<Vector> globalAveragePoolingGPU(
    GPUTensor<Matrix> input, CommandBuffer tape, AllocBuffer allocTape) {
  int dModel = input.shape[1];

  List<int> outShape = <int>[dModel];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape, allocTape,
      sourceOp: 'global_avg_pool_gpu');

  tape.putInt(OP_GLOBAL_AVG_POOL_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_GLOBAL_AVG_POOL_BACKWARD);
      backwardTape.putString(out.gradId);
      backwardTape.putString(input.gradId);
    },
    opName: 'global_avg_pool_gpu',
  );

  return out;
}

/// Normalizes a 1D feature vector using per-feature running mean and variance.
/// Scales and shifts the normalized data using learnable [gamma] and [beta] parameter vectors.
/// During inference, it computes the output strictly using the pre-calculated running statistics rather than the current batch's statistics.
/// The operations are appended to [tape] for execution.
GPUTensor<Vector> batchNorm1dGPU(
    GPUTensor<Vector> input,
    GPUTensor<Vector> gamma,
    GPUTensor<Vector> beta,
    GPUTensor<Vector> runningMean,
    GPUTensor<Vector> runningVariance,
    double momentum,
    double epsilon,
    bool isTraining,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int numFeatures = input.shape[0];

  GPUTensor<Vector> out = GPUTensor<Vector>.empty(
      <int>[numFeatures], allocTape,
      sourceOp: 'batchNorm1dGPU');
  GPUTensor<Vector> savedMean = GPUTensor<Vector>.empty(
      <int>[numFeatures], allocTape,
      requiresGrad: false, sourceOp: 'batchNorm1dGPU:savedMean');
  GPUTensor<Vector> savedInvVar = GPUTensor<Vector>.empty(
      <int>[numFeatures], allocTape,
      requiresGrad: false, sourceOp: 'batchNorm1dGPU:savedInvVar');
  out.subMap["savedMean"] = savedMean;
  out.subMap["savedInvVar"] = savedInvVar;

  tape.putInt(OP_BATCH_NORM_1D_FORWARD);
  tape.putString(input.id);
  tape.putString(gamma.id);
  tape.putString(beta.id);
  tape.putString(runningMean.id);
  tape.putString(runningVariance.id);
  tape.putString(out.id);
  tape.putString(savedMean.id);
  tape.putString(savedInvVar.id);
  tape.putFloat(momentum);
  tape.putFloat(epsilon);
  tape.putPatchableBool('is_training', isTraining);

  out.creator = GPUNode(
    <GPUTensor>[input, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BATCH_NORM_1D_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.id);
      bTape.putString(gamma.id);
      bTape.putString(savedMean.id);
      bTape.putString(savedInvVar.id);
      bTape.putString(input.gradId);
      bTape.putString(gamma.gradId);
      bTape.putString(beta.gradId);
    },
    opName: 'batchNorm1dGPU',
    cost: numFeatures * 4,
  );

  return out;
}

/// Normalizes a 3D spatial tensor independently for each channel.
/// Uses the mean and variance evaluated across the combined spatial dimensions (height and width) of a given channel.
/// Returns a 3D tensor of identical shape scaled by [gamma] and shifted by [beta].
/// The operations are appended to [tape] for execution.
GPUTensor<Tensor3D> batchNorm2dGPU(
    GPUTensor<Tensor3D> input,
    GPUTensor<Vector> gamma,
    GPUTensor<Vector> beta,
    GPUTensor<Vector> runningMean,
    GPUTensor<Vector> runningVariance,
    double momentum,
    double epsilon,
    bool isTraining,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int numChannels = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[numChannels, height, width], allocTape,
      sourceOp: 'batchNorm2dGPU');
  GPUTensor<Vector> savedMean = GPUTensor<Vector>.empty(
      <int>[numChannels], allocTape,
      requiresGrad: false, sourceOp: 'batchNorm2dGPU:savedMean');
  GPUTensor<Vector> savedInvVar = GPUTensor<Vector>.empty(
      <int>[numChannels], allocTape,
      requiresGrad: false, sourceOp: 'batchNorm2dGPU:savedInvVar');
  out.subMap["savedMean"] = savedMean;
  out.subMap["savedInvVar"] = savedInvVar;

  tape.putInt(OP_BATCH_NORM_2D_FORWARD);
  tape.putString(input.id);
  tape.putString(gamma.id);
  tape.putString(beta.id);
  tape.putString(runningMean.id);
  tape.putString(runningVariance.id);
  tape.putString(out.id);
  tape.putString(savedMean.id);
  tape.putString(savedInvVar.id);
  tape.putFloat(momentum);
  tape.putFloat(epsilon);
  tape.putPatchableBool('is_training', isTraining);

  out.creator = GPUNode(
    <GPUTensor>[input, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BATCH_NORM_2D_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.id);
      bTape.putString(gamma.id);
      bTape.putString(savedMean.id);
      bTape.putString(savedInvVar.id);
      bTape.putString(input.gradId);
      bTape.putString(gamma.gradId);
      bTape.putString(beta.gradId);
    },
    opName: 'batchNorm2dGPU',
    cost: numChannels * height * width * 4,
  );

  return out;
}

/// Normalizes a 2D matrix independently for each row.
/// Calculates the mean and variance across the columns of a single row, normalizes the row's elements, and applies the [gamma] and [beta] transformations.
/// Outputs a normalized matrix of the original dimensions.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> layerNormMatrixGPU(
    GPUTensor<Matrix> m,
    GPUTensor<Vector> gamma,
    GPUTensor<Vector> beta,
    GPUTensor<Vector> meanCache,
    GPUTensor<Vector> rstdCache,
    double epsilon,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(
      <int>[numRows, numCols], allocTape,
      sourceOp: 'layerNormMatrixGPU');

  tape.putInt(OP_LAYER_NORM_FORWARD);
  tape.putString(m.id);
  tape.putString(gamma.id);
  tape.putString(beta.id);
  tape.putString(out.id);
  tape.putString(meanCache.id);
  tape.putString(rstdCache.id);
  tape.putFloat(epsilon);

  out.creator = GPUNode(
    <GPUTensor>[m, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_LAYER_NORM_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(m.id);
      bTape.putString(gamma.id);
      bTape.putString(meanCache.id);
      bTape.putString(rstdCache.id);
      bTape.putString(m.gradId);
      bTape.putString(gamma.gradId);
      bTape.putString(beta.gradId);
    },
    opName: 'layerNormMatrixGPU',
    cost: numRows * numCols * 8,
  );

  return out;
}

/// Randomly zeroes out elements of the input tensor with a probability equal to the specified [rate].
/// To preserve the expected mathematical sum of the tensor during inference, it scales the remaining active elements by 1 / (1 - rate).
/// Outputs the modified tensor alongside the generated dropout mask.
/// The operations are appended to [tape] for execution.
GPUTensor<T> dropoutGPU<T>(GPUTensor<T> input, double rate, CommandBuffer tape,
    AllocBuffer allocTape) {
  int seed = Random().nextInt(1000000);

  GPUTensor<T> out =
  GPUTensor<T>.empty(input.shape, allocTape, sourceOp: 'dropout_gpu');
  GPUTensor<T> mask = GPUTensor<T>.empty(input.shape, allocTape,
      requiresGrad: false, sourceOp: 'dropout_gpu:mask');
  out.subMap["mask"] = mask;

  tape.putInt(OP_DROPOUT_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(mask.id);
  tape.putPatchableFloat('dropout_rate', rate);
  tape.putPatchableInt('dropout_seed', seed);

  int elementCount = 1;
  for (int i = 0; i < input.shape.length; i = i + 1) {
    elementCount = elementCount * input.shape[i];
  }

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DROPOUT_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(mask.id);
      bTape.putString(input.gradId);
    },
    opName: 'dropout_gpu',
    cost: elementCount * 2,
  );

  return out;
}

/// Constructs a Markov chain probability transition table from a given state [sequence].
/// Calculates the frequencies of state transitions based on the specified Markov [order] and [numStates],
/// and normalizes these counts to output a probability matrix of shape `[numStates^order, numStates]`. This Table does not have a gradient calculation.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> buildMarkovTableGPU(GPUTensor<Vector> sequence, int order,
    int numStates, CommandBuffer tape, AllocBuffer allocTape) {
  int numHistories = math.pow(numStates, order).toInt();

  GPUTensor<Matrix> countTable = GPUTensor<Matrix>.empty(
      <int>[numHistories, numStates], allocTape,
      requiresGrad: false, sourceOp: 'buildMarkovTableGPU:countTable');
  GPUTensor<Matrix> probTable = GPUTensor<Matrix>.empty(
      <int>[numHistories, numStates], allocTape,
      requiresGrad: false, sourceOp: 'buildMarkovTableGPU');
  probTable.subMap["countTable"] = countTable;

  tape.putInt(OP_FILL);
  tape.putString(countTable.id);
  tape.putFloat(0.0);

  tape.putInt(OP_MARKOV_COUNT);
  tape.putString(sequence.id);
  tape.putString(countTable.id);
  tape.putInt(order);
  tape.putInt(numStates);

  tape.putInt(OP_MARKOV_NORMALIZE);
  tape.putString(countTable.id);
  tape.putString(probTable.id);
  tape.putInt(numHistories);
  tape.putInt(numStates);

  probTable.creator = GPUNode(<GPUTensor>[sequence], (CommandBuffer bTape) {},
      opName: 'buildMarkovTableGPU');

  return probTable;
}

/// Predicts the probability distribution of the next states for a batch of histories.
/// Maps each sequence in the [historyBatch] to its corresponding row in the provided [probTable]
/// and outputs a matrix of shape `[batchSize, numStates]` containing the next-state probabilities.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> markovPredictGPU(
    GPUTensor<Matrix> historyBatch,
    GPUTensor<Matrix> probTable,
    int numStates,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int batchSize = historyBatch.shape[0];
  int order = historyBatch.shape[1];

  GPUTensor<Matrix> outProbs = GPUTensor<Matrix>.empty(
      <int>[batchSize, numStates], allocTape,
      requiresGrad: false, sourceOp: 'markovPredictGPU');

  tape.putInt(OP_MARKOV_PREDICT);
  tape.putString(historyBatch.id);
  tape.putString(probTable.id);
  tape.putString(outProbs.id);
  tape.putInt(order);
  tape.putInt(numStates);

  outProbs.creator = GPUNode(
      <GPUTensor>[historyBatch, probTable], (CommandBuffer bTape) {},
      opName: 'markovPredictGPU');

  return outProbs;
}

/// /////////////////////////////////
/// Fused Kernels (1000+)         ///
/// /////////////////////////////////

/// Executes a fused matrix multiplication, bias addition, and ReLU activation sequence.
/// Multiplies matrix [x] by weight matrix [w], broadcasts and adds the bias vector [b],
/// and applies a ReLU activation in a single optimized pass.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> matMulBiasReluGPU(
    GPUTensor<Matrix> x,
    GPUTensor<Matrix> w,
    GPUTensor<Vector> b,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int M = x.shape[0];
  int K = x.shape[1];
  int N = w.shape[1];

  List<int> outShape = <int>[M, N];
  GPUTensor<Matrix> reluOut = GPUTensor<Matrix>.empty(outShape, allocTape,
      sourceOp: 'matMulBiasReluGPU');
  GPUTensor<Matrix> preReluOut = GPUTensor<Matrix>.empty(outShape, allocTape,
      sourceOp: 'matMulBiasReluGPU:preReluOut');

  reluOut.subMap["preReluOut"] = preReluOut;

  tape.putInt(OP_MATMUL_BIAS_RELU_FORWARD);
  tape.putString(x.id);
  tape.putString(w.id);
  tape.putString(b.id);
  tape.putString(reluOut.id);
  tape.putString(preReluOut.id);

  int cost = (2 * M * K * N) + (M * N) + (M * N);

  reluOut.creator = GPUNode(
    <GPUTensor>[x, w, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_RELU_BACKWARD);
      bTape.putString(preReluOut.id);
      bTape.putString(reluOut.gradId);
      bTape.putString(preReluOut.gradId);

      bTape.putInt(OP_MATMUL);
      bTape.putString(x.id);
      bTape.putString(preReluOut.gradId);
      bTape.putString(w.gradId);
      bTape.putBool(true);
      bTape.putBool(false);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true);

      bTape.putInt(OP_MATMUL);
      bTape.putString(preReluOut.gradId);
      bTape.putString(w.id);
      bTape.putString(x.gradId);
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true);

      bTape.putInt(OP_SUM_REDUCE_COLUMNS);
      bTape.putString(preReluOut.gradId);
      bTape.putString(b.gradId);
    },
    opName: 'matMulBiasReluGPU',
    cost: cost,
  );

  return reluOut;
}

/// Calculates the dot product between vector [a] and [b].
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> dotProductGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> multiplied = multiplyGPU<Vector>(a, b, tape, allocTape);
  GPUTensor<Scalar> out = sumGPU(multiplied, tape, allocTape);
  out.subMap["multiplied"] = multiplied;
  return out;
}

/// Computes the L2 norm (Euclidean length) of a vector. Squares each element of the vector, sums them together, and takes the square root of the result.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> l2NormGPU(
    GPUTensor<Vector> v, CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> squared = powGPU<Vector>(v, 2.0, tape, allocTape);
  GPUTensor<Scalar> sumOfSquares = sumGPU(squared, tape, allocTape);
  GPUTensor<Scalar> out = sqrtGPU<Scalar>(sumOfSquares, tape, allocTape);

  out.subMap["squared"] = squared;
  out.subMap["sumOfSquares"] = sumOfSquares;
  return out;
}

/// Calculates the straight-line Euclidean distance between two vectors.
/// Computes the element-wise difference between vector [a] and [b], squares the differences, sums them, and extracts the square root.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> euclideanDistanceGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> diff = subtractGPU<Vector>(a, b, tape, allocTape);
  GPUTensor<Vector> squaredDiff = powGPU<Vector>(diff, 2.0, tape, allocTape);
  GPUTensor<Scalar> sumOfSquares = sumGPU(squaredDiff, tape, allocTape);
  GPUTensor<Scalar> out = sqrtGPU<Scalar>(sumOfSquares, tape, allocTape);
  out.subMap["diff"] = diff;
  out.subMap["squaredDiff"] = squaredDiff;
  out.subMap["sumOfSquares"] = sumOfSquares;
  return out;
}

/// Computes the cosine similarity between two vectors, measuring their directional alignment.
/// Calculates the dot product of [a] and [b] and divides it by the product of their individual L2 norms.
/// Returns a scalar between -1.0 (opposite) and 1.0 (identical direction).
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> cosineSimilarityGPU(GPUTensor<Vector> a, GPUTensor<Vector> b,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Scalar> dot = dotProductGPU(a, b, tape, allocTape);
  GPUTensor<Scalar> normA = l2NormGPU(a, tape, allocTape);
  GPUTensor<Scalar> normB = l2NormGPU(b, tape, allocTape);
  GPUTensor<Scalar> denominator =
  multiplyGPU<Scalar>(normA, normB, tape, allocTape);
  GPUTensor<Scalar> out = divideGPU<Scalar>(dot, denominator, tape, allocTape);
  out.subMap["dot"] = dot;
  out.subMap["normA"] = normA;
  out.subMap["normB"] = normB;
  out.subMap["denominator"] = denominator;
  return out;
}

/// Calculates the Mean Absolute Error (MAE) loss between predictions and targets.
/// Computes the absolute differences between the [preds] and [targets] vectors, sums the errors, and divides by the total number of elements to yield a single average error scalar.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> maeLossGPU(GPUTensor<Vector> preds, GPUTensor<Vector> targets,
    CommandBuffer tape, AllocBuffer allocTape) {
  GPUTensor<Vector> diff = subtractGPU<Vector>(preds, targets, tape, allocTape);
  GPUTensor<Vector> absoluteDiff = absGPU<Vector>(diff, tape, allocTape);
  GPUTensor<Scalar> totalError = sumGPU(absoluteDiff, tape, allocTape);
  GPUTensor<Scalar> nScalar = GPUTensor<Scalar>(
      preds.shape[0].toDouble(), allocTape,
      sourceOp: 'maeLossGPU:nScalar');
  GPUTensor<Scalar> out =
  divideGPU<Scalar>(totalError, nScalar, tape, allocTape);
  out.subMap["diff"] = diff;
  out.subMap["absoluteDiff"] = absoluteDiff;
  out.subMap["totalError"] = totalError;
  out.subMap["nScalar"] = nScalar;
  return out;
}

/// /////////////////////////////////
/// Transformer & LLM Ops (2000+) ///
/// /////////////////////////////////

/// Applies Root Mean Square (RMS) Normalization across the feature dimension of an input matrix.
/// Optionally scales the normalized features using a learnable [weight] vector. If [scalePlusOne] is true (as required by Gemma models), the scaling factor applied is `(1.0 + weight)`. The reciprocal standard deviation is cached in [rstdCache] to accelerate backpropagation.
GPUTensor<Matrix> rmsNormMatrixGPU(
    GPUTensor<Matrix> input,
    GPUTensor<Vector>? weight,
    double epsilon,
    bool scalePlusOne,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int numRows = input.shape[0];
  int numCols = input.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols],
      allocTape,
      sourceOp: 'rmsNormMatrixGPU');
  GPUTensor<Vector> rstdCache = GPUTensor<Vector>.empty(<int>[numRows],
      allocTape,
      requiresGrad: false, sourceOp: 'rmsNormMatrixGPU:rstdCache');
  out.subMap["rstdCache"] = rstdCache;

  String weightId = '';
  String weightGradId = '';

  if (weight != null) {
    weightId = weight.id;
    weightGradId = weight.gradId;
  }

  tape.putInt(OP_RMS_NORM_FORWARD);
  tape.putString(input.id);
  tape.putString(weightId);
  tape.putString(out.id);
  tape.putString(rstdCache.id);
  tape.putFloat(epsilon);
  tape.putBool(scalePlusOne);

  List<GPUTensor> inputs = <GPUTensor>[input];
  if (weight != null) {
    inputs.add(weight);
  }

  out.creator = GPUNode(
    inputs,
        (CommandBuffer bTape) {
      bTape.putInt(OP_RMS_NORM_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.id);
      bTape.putString(weightId);
      bTape.putString(rstdCache.id);
      bTape.putString(input.gradId);
      bTape.putString(weightGradId);
      bTape.putBool(scalePlusOne);
    },
    opName: 'rmsNormMatrixGPU',
    cost: numRows * numCols * 4,
  );

  return out;
}

/// Applies a causal attention mask to prevent sequences from attending to future tokens.
/// Modifies the `[seqLen, seqLen]` attention matrices within the 3D tensor by overriding elements
/// above the main diagonal with `-10000.0`, effectively zeroing out their probability post-softmax.
/// Gradients are blocked from flowing back through the masked positions.
GPUTensor<Tensor3D> causalMaskGPU(GPUTensor<Tensor3D> input, int batchSize,
    int numHeads, int seqLen, CommandBuffer tape, AllocBuffer allocTape) {
  int depth = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'causalMaskGPU');

  tape.putInt(OP_CAUSAL_MASK_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(batchSize);
  tape.putInt(numHeads);
  tape.putInt(seqLen);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CAUSAL_MASK_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(input.gradId);
      bTape.putInt(batchSize);
      bTape.putInt(numHeads);
      bTape.putInt(seqLen);
    },
    opName: 'causalMaskGPU',
    cost: batchSize * numHeads * seqLen * seqLen,
  );

  return out;
}

/// Applies Rotary Position Embeddings (RoPE) to a multi-head attention state tensor.
/// Encodes relative positional information by pairing and rotating feature dimensions using precomputed values supplied in [cosTable] and [sinTable]. The backward pass computes the
/// inverse rotation gradients by intrinsically transposing the rotation matrix (negating the sine terms).
GPUTensor<Tensor3D> applyRopeGPU(
    GPUTensor<Tensor3D> input,
    GPUTensor<Matrix> cosTable,
    GPUTensor<Matrix> sinTable,
    int batchSize,
    int seqLen,
    int numHeads,
    int headDim,
    CommandBuffer tape,
    AllocBuffer allocTape) {
  int depth = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(
      <int>[depth, height, width], allocTape,
      sourceOp: 'applyRopeGPU');

  tape.putInt(OP_ROPE_FORWARD);
  tape.putString(input.id);
  tape.putString(cosTable.id);
  tape.putString(sinTable.id);
  tape.putString(out.id);
  tape.putInt(batchSize);
  tape.putInt(seqLen);
  tape.putInt(numHeads);
  tape.putInt(headDim);

  out.creator = GPUNode(
    <GPUTensor>[input, cosTable, sinTable],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ROPE_BACKWARD);
      bTape.putString(out.gradId);
      bTape.putString(cosTable.id);
      bTape.putString(sinTable.id);
      bTape.putString(input.gradId);
      bTape.putInt(batchSize);
      bTape.putInt(seqLen);
      bTape.putInt(numHeads);
      bTape.putInt(headDim);
    },
    opName: 'applyRopeGPU',
    cost: batchSize * seqLen * numHeads * headDim,
  );

  return out;
}

/// Computes the categorical Cross-Entropy loss between predicted logits and target class indices.
/// Leverages a fused, numerically stable softmax internally to calculate the log-probabilities.
/// Outputs an unreduced vector containing one scalar loss value per batch sequence.
GPUTensor<Vector> crossEntropyLossGPU(GPUTensor<Matrix> logits,
    GPUTensor<Vector> targets, CommandBuffer tape, AllocBuffer allocTape) {
  int batchSize = logits.shape[0];
  int vocabSize = logits.shape[1];

  GPUTensor<Vector> outLoss = GPUTensor<Vector>.empty(<int>[batchSize],
      allocTape,
      sourceOp: 'crossEntropyLossGPU');

  tape.putInt(OP_CROSS_ENTROPY_FORWARD);
  tape.putString(logits.id);
  tape.putString(targets.id);
  tape.putString(outLoss.id);

  outLoss.creator = GPUNode(
    <GPUTensor>[logits, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CROSS_ENTROPY_BACKWARD);
      bTape.putString(outLoss.gradId);
      bTape.putString(logits.id);
      bTape.putString(targets.id);
      bTape.putString(logits.gradId);
    },
    opName: 'crossEntropyLossGPU',
    cost: batchSize * vocabSize,
  );

  return outLoss;
}

/// Identifies the predicted class token for each sequence in a batch.
/// Examines the vocabulary dimension of the provided [logits] and returns a vector containing
/// the index of the maximum value (argmax) for each batch row. This operator is used exclusively
/// during inference and has no backward pass.
GPUTensor<Vector> argmaxGPU(
    GPUTensor<Matrix> logits, CommandBuffer tape, AllocBuffer allocTape) {
  int batchSize = logits.shape[0];
  int vocabSize = logits.shape[1];

  GPUTensor<Vector> outIndices = GPUTensor<Vector>.empty(
      <int>[batchSize], allocTape,
      requiresGrad: false, sourceOp: 'argmaxGPU');

  tape.putInt(OP_ARGMAX_FORWARD);
  tape.putString(logits.id);
  tape.putString(outIndices.id);

  outIndices.creator = GPUNode(
    <GPUTensor>[logits],
        (CommandBuffer bTape) {},
    opName: 'argmaxGPU',
    cost: batchSize * vocabSize,
  );

  return outIndices;
}