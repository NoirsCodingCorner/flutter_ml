import 'dart:math';
import 'dart:math' as math;
import 'tensor_gpu.dart';
import 'type_Aliases.dart';

import '../gpu_version/ffi/OpCodes.dart';
import '../gpu_version/ffi/commandBuffer.dart';


/// /////////////////////////////////
/// Data management (0-99)         ///
/// /////////////////////////////////

GPUTensor<Matrix> reshapeVectorToMatrixGPU(GPUTensor<Vector> v, int numRows, int numCols, CommandBuffer tape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_COPY);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'reshapeGPU',
    extraParams: {'numRows': numRows, 'numCols': numCols},
    cost: 0,
  );

  return out;
}
GPUTensor<Tensor3D> reshapeMatrixTo3DGPU(GPUTensor<Matrix> m, int c, int h, int w, CommandBuffer tape) {
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[c, h, w]);

  tape.putInt(OP_COPY);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'reshapeMatrixTo3DGPU',
    cost: 0,
  );

  return out;
}
GPUTensor<Matrix> reshape3DToMatrixGPU(GPUTensor<Tensor3D> t, int rows, int cols, CommandBuffer tape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[rows, cols]);

  tape.putInt(OP_COPY);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${t.id}_grad');
    },
    opName: 'reshape3DToMatrixGPU',
    cost: 0,
  );

  return out;
}
GPUTensor<Matrix> flatten3DToMatrixGPU(GPUTensor<Tensor3D> t, CommandBuffer tape) {
  int c = t.shape[0];
  int h = t.shape[1];
  int w = t.shape[2];
  int flatSize = c * h * w;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[1, flatSize]);

  tape.putInt(OP_COPY);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${t.id}_grad');
    },
    opName: 'flatten3DToMatrixGPU',
    cost: 0,
  );

  return out;
}

GPUTensor<Vector> loadSampleGPU(GPUTensor<Matrix> dataset, int sampleIndex, CommandBuffer tape) {
  int cols = dataset.shape[1];

  List<int> outShape = <int>[cols];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape);

  tape.putInt(OP_LOAD_SAMPLE);
  tape.putString(dataset.id);
  tape.putString(out.id);
  tape.putInt(sampleIndex);

  out.creator = GPUNode(
    <GPUTensor>[dataset],
        (CommandBuffer bTape) {
      bTape.putInt(OP_STORE_SAMPLE);
      bTape.putString('${out.id}_grad');
      bTape.putString('${dataset.id}_grad');
      bTape.putInt(sampleIndex);
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
GPUTensor<T> addGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape) {
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'addGPU',
  );
  return out;
}
/// Adds the command for addition of two vectors [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Vector> addVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(a.shape[0], 0.0));

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'addVectorGPU',
  );

  return out;
}
/// Adds the command for addition of two matrices [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> addMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  // Initialize with an empty structure to set the shape
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'add_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}
/// Adds the command for addition of two Tensor3Ds [a] and [b] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> add3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[depth, height, width]);

  tape.putInt(OP_ADD);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'add_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for subtraction of two GPUTensors [a] and [b] of type [T] to the tape.
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> subtractGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape) {
  // Correctly inherit shape to pre-allocate VRAM
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      // Forward: C = A - B
      // Backward: dA += dC, dB -= dC
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'subtractGPU',
  );

  return out;
}GPUTensor<Vector> subtractVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  int length = a.shape[0];
  List<int> shape = <int>[length];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(shape);

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'subtractVectorGPU',
    cost: length,
  );

  return out;
}
/// Appends the commands for subtraction of two matrices [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> subtractMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];
  List<int> shape = <int>[numRows, numCols];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(shape);

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'subtractMatrixGPU',
    cost: numRows * numCols,
  );

  return out;
}
/// Appends the commands for subtraction of two Tensor3Ds [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> subtract3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];
  List<int> shape = <int>[depth, height, width];
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(shape);

  tape.putInt(OP_SUBTRACT);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_SUBTRACT_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'subtract3DGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends the commands for element wise multiplication of two GPUTensors [a] and [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> multiplyGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape) {
  // Correctly inherit shape to pre-allocate VRAM
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${b.id}_grad');
    },
    opName: 'multiplyGPU',
    cost: 1,
  );

  return out;
}GPUTensor<Scalar> multiplyScalarGPU(GPUTensor<Scalar> a, GPUTensor<Scalar> b, CommandBuffer tape) {
  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${b.id}_grad');
    },
    opName: 'multiplyScalarGPU',
    cost: 1,
  );

  return out;
}
/// Appends the commands for element wise multiplication of two vectors [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Vector> elementWiseMultiplyGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(a.shape[0], 0.0));

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${b.id}_grad');
    },
    opName: 'elementWiseMultiplyGPU',
    cost: 1,
  );

  return out;
}
/// Appends the commands for element wise multiplication of two Tensor3Ds [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> elementWiseMultiply3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty([depth, height, width]);

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${b.id}_grad');
    },
    opName: 'multiply_3dGPU',
    cost: depth * height * width,
  );

  return out;
}
/// Appends the commands for element wise multiplication of two matrices [a] and [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> elementWiseMultiplyMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([numRows, numCols]);

  tape.putInt(OP_MULTIPLY);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');

      bTape.putInt(OP_MULTIPLY_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${b.id}_grad');
    },
    opName: 'multiply_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends the commands for element wise division of two GPUTensors [a] through [b] of type [T] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<T> divideGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape) {
  // Correctly inherit shape to pre-allocate VRAM
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

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
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'divideGPU',
  );

  return out;
}GPUTensor<Vector> divideVectorGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  int length = a.shape[0];
  List<int> shape = <int>[length];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(shape);

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
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'divideVectorGPU',
    cost: length,
  );

  return out;
}
/// Appends the commands for element wise division of two matrices [a] through [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Matrix> divideMatrixGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];
  List<int> shape = <int>[numRows, numCols];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(shape);

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
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'divideMatrixGPU',
    cost: numRows * numCols,
  );

  return out;
}
/// Appends the commands for element wise division of two Tensor3Ds [a] through [b] to [tape].
/// Creates and allocates a Tensor to store the results.
GPUTensor<Tensor3D> divide3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];
  List<int> shape = <int>[depth, height, width];
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(shape);

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
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'divide3DGPU',
    cost: depth * height * width,
  );

  return out;
}


/// Appends the commands for applying the e^v exponential function on vector [v] to [tape].
GPUTensor<Vector> vectorExpGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  int N = v.shape[0];
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(N, 0.0));

  tape.putInt(OP_EXP_ELEMENTWISE);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EXP_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.id);
      bTape.putString(v.id);
    },
    opName: 'exp_vectorGPU',
    cost: N,
  );

  return out;
}

/// Appends the commands for applying the abs(v) exponential function on every element of [v] to [tape].
GPUTensor<T> absGPU<T>(GPUTensor<T> a, CommandBuffer tape) {
  // CORRECT: Inherits the exact shape from the input tensor (no 0-length tensors)
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_ABS_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ABS_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${a.id}_grad');
    },
    opName: 'absGPU',
  );

  return out;
}

/// Appends the commands for applying the sqrt(v) exponential function on every element of [v] to [tape].
GPUTensor<T> sqrtGPU<T>(GPUTensor<T> a, CommandBuffer tape) {
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_SQRT_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      // Sqrt backward optimizes perfectly by passing the OUT data instead of IN
      bTape.putInt(OP_SQRT_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(out.id);
      bTape.putString('${a.id}_grad');
    },
    opName: 'sqrtGPU',
  );

  return out;
}
/// Appends the commands for applying the log(v) exponential function on every element of [v] to [tape].
GPUTensor<T> logGPU<T>(GPUTensor<T> a, CommandBuffer tape) {
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_LOG_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_LOG_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${a.id}_grad');
    },
    opName: 'logGPU',
  );

  return out;
}
/// Appends the commands for applying pow([v],[exponent]) exponential function on every element of [v] to [tape].
GPUTensor<T> powGPU<T>(GPUTensor<T> a, double exponent, CommandBuffer tape) {
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_POW_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);
  tape.putFloat(exponent);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_POW_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${a.id}_grad');
      bTape.putFloat(exponent);
    },
    opName: 'powGPU',
  );

  return out;
}
/// Appends the commands for clamping every element of [v] between [minVal] - [maxVal] to [tape].
GPUTensor<T> clampGPU<T>(GPUTensor<T> a, double minVal, double maxVal, CommandBuffer tape) {
  GPUTensor<T> out = GPUTensor<T>.empty(a.shape);

  tape.putInt(OP_CLAMP_ELEMENTWISE);
  tape.putString(a.id);
  tape.putString(out.id);
  tape.putFloat(minVal);
  tape.putFloat(maxVal);

  out.creator = GPUNode(
    <GPUTensor>[a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CLAMP_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(a.id);
      bTape.putString('${a.id}_grad');
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
GPUTensor<Matrix> matMulGPU(GPUTensor<Matrix> a, GPUTensor<Matrix> b, CommandBuffer tape) {
  int M = a.shape[0];
  int N = a.shape[1];
  int P = b.shape[1];

  // Instantly reserve VRAM based on shape, bypassing the Dart Heap
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[M, P]);

  tape.putInt(OP_MATMUL);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putBool(false);
  tape.putBool(false);
  tape.putFloat(1.0);
  tape.putFloat(0.0);
  tape.putBool(true); // <--- Tensor Cores ON for Forward Pass

  int cost = 2 * M * N * P;

  out.creator = GPUNode(
    <GPUTensor>[a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MATMUL);
      bTape.putString('${out.id}_grad');
      bTape.putString(b.id);
      bTape.putString('${a.id}_grad');
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true);

      bTape.putInt(OP_MATMUL);
      bTape.putString(a.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
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
GPUTensor<Vector> matVecMulGPU(GPUTensor<Matrix> mMat, GPUTensor<Vector> v, CommandBuffer tape) {
  int numRows = mMat.shape[0];
  int numCols = mMat.shape[1];

  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(numRows, 0.0));

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
    [mMat, v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MATMUL);
      bTape.putString('${out.id}_grad');
      bTape.putString(v.id);
      bTape.putString('${mMat.id}_grad');
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(false);

      bTape.putInt(OP_MATMUL);
      bTape.putString(mMat.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
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
GPUTensor<Matrix> transposeGPU(GPUTensor<Matrix> a, CommandBuffer tape) {
  int M = a.shape[0];
  int N = a.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[N, M]);
  // tmpGrad must match original input shape [M, N]
  GPUTensor<Matrix> tmpGrad = GPUTensor<Matrix>.empty(<int>[M, N]);

  tape.putInt(OP_TRANSPOSE);
  tape.putString(a.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [a],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TRANSPOSE);
      bTape.putString('${out.id}_grad');
      bTape.putString(tmpGrad.id);

      bTape.putInt(OP_ADD_INTO);
      bTape.putString(tmpGrad.id);
      bTape.putString('${a.id}_grad');
    },
    opName: 'transposeGPU',
    extraParams: {'tmpGrad': tmpGrad},
    cost: 0,
  );

  return out;
}

/// Appends a command for broadcast add a vector [v] to [tape].
GPUTensor<Matrix> addMatrixAndVectorGPU(GPUTensor<Matrix> m, GPUTensor<Vector> v, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m, v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');

      bTape.putInt(OP_SUM_REDUCE_COLUMNS);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'addMatrixAndVectorGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends a command for adding a scalar [b] to every element of a matrix [m] to [tape].
GPUTensor<Matrix> addScalarMatrixGPU(GPUTensor<Matrix> m, GPUTensor<Scalar> b, CommandBuffer tape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  List<int> shape = <int>[rows, cols];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(shape);

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');

      bTape.putInt(OP_SUM_REDUCE);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'addScalarToMatrixGPU',
    cost: rows * cols,
  );

  return out;
}
/// Appends a command for adding a scalar [b] to every element of a vector [v] to [tape].
GPUTensor<Vector> addScalarVectorGPU(GPUTensor<Vector> v, double scalar, CommandBuffer tape) {
  int length = v.shape[0];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(<int>[length]);

  tape.putInt(OP_ADD_SCALAR);
  tape.putString(v.id);
  tape.putString(out.id);
  tape.putFloat(scalar);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      // The derivative of (x + c) is 1, so the gradient passes directly through
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'addScalarVectorGPU',
    cost: length,
  );

  return out;
}
/// Appends a command for adding a scalar [b] to every element of a Tensor3D [v] to [tape].
GPUTensor<Tensor3D> addScalar3DGPU(GPUTensor<Tensor3D> t, double scalar, CommandBuffer tape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[depth, height, width]);

  tape.putInt(OP_ADD_SCALAR);
  tape.putString(t.id);
  tape.putString(out.id);
  tape.putFloat(scalar);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${t.id}_grad');
    },
    opName: 'addScalar3DGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for broadcast additions of a matrix wrapped scalar [b] to a full matrix [m].
GPUTensor<Matrix> addBiasToFeatureMapGPU(GPUTensor<Matrix> m, GPUTensor<Matrix> b, CommandBuffer tape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  List<int> shape = <int>[rows, cols];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(shape);

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');

      bTape.putInt(OP_SUM_REDUCE);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'addBiasToFeatureMapGPU',
    cost: rows * cols,
  );

  return out;
}
/// Appends commands for addition of a 1D bias vector [b] to a 2D matrix [m] via broadcasting over the matrix to the [tape].
GPUTensor<Matrix> addBiasToMatMulOutGPU(GPUTensor<Matrix> m, GPUTensor<Vector> b, CommandBuffer tape) {
  int rows = m.shape[0];
  int cols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[rows, cols]);

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(b.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_ADD_INTO);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');

      // FIXED: Sum across columns to isolate the row biases!
      bTape.putInt(OP_SUM_REDUCE_ROWS);
      bTape.putString('${out.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'addBiasToMatMulOutGPU',
    cost: rows * cols,
  );

  return out;
}
/// Appends commands for addition of a 1D bias vector [b] to a 2D matrix [m] via broadcasting over the matrix to the [tape].
/// This command is inference only! No gradient path is calculated.
GPUTensor<Matrix> broadcastAddVectorToMatrixGPU(GPUTensor<Matrix> m, GPUTensor<Vector> v, CommandBuffer tape) {
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(m.shape);

  tape.putInt(OP_BROADCAST_ADD);
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m, v],
        (CommandBuffer bTape) {
      // Backward pass logic (stubbed for inference)
    },
    opName: 'broadcastAddVectorToMatrixGPU',
    cost: m.shape[0] * m.shape[1],
  );

  return out;
}
/// Appends commands for additions of a scalar constant [s] to every element of matrix [m] to [tape].
GPUTensor<Matrix> scaleMatrixGPU(GPUTensor<Matrix> m, double s, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_SCALE_MATRIX);
  tape.putString(m.id);
  tape.putString(out.id);
  tape.putFloat(s);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SCALE_MATRIX_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
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
GPUTensor<Vector> reluGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(v.shape[0], 0.0));

  tape.putInt(OP_RELU);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_RELU_BACKWARD);
      bTape.putString(v.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'reluGPU',
    cost: v.shape[0],
  );

  return out;
}
/// Appends commands for applying Relu on every element of matrix [m] to [tape].
GPUTensor<Matrix> reluMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_RELU);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_RELU_BACKWARD);
      bTape.putString(m.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'relu_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the sigmoid function on every element of [m] (type [t]) to [tape].
GPUTensor<T> sigmoidScalarGPU<T>(GPUTensor<T> s, CommandBuffer tape) {
  dynamic dummy = T == Scalar ? 0.0 : (T == Vector ? <double>[] : <List<double>>[]);
  GPUTensor<T> out = GPUTensor<T>(dummy);

  tape.putInt(OP_SIGMOID);
  tape.putString(s.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [s],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${s.id}_grad');
    },
    opName: 'sigmoidScalarGPU',
    cost: 1,
  );

  return out;
}
/// Appends commands for applying the sigmoid function on every element of vector [v] to [tape].
GPUTensor<Vector> sigmoidGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(v.shape[0], 0.0));

  tape.putInt(OP_SIGMOID);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'sigmoidGPU',
    cost: v.shape[0],
  );

  return out;
}
/// Appends commands for applying the sigmoid function on every element of matrix [m] to [tape].
GPUTensor<Matrix> sigmoidMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_SIGMOID);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'sigmoid_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}
/// Appends commands for applying the sigmoid function on every element of Tensor3D [t] to [tape].
GPUTensor<Tensor3D> sigmoid3DGPU(GPUTensor<Tensor3D> t, CommandBuffer tape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];

  List<List<List<double>>> zeros = <List<List<double>>>[];
  for (int c = 0; c < depth; c = c + 1) {
    List<List<double>> channel = <List<double>>[];
    for (int h = 0; h < height; h = h + 1) {
      List<double> row = <double>[];
      for (int w = 0; w < width; w = w + 1) {
        row.add(0.0);
      }
      channel.add(row);
    }
    zeros.add(channel);
  }

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>(zeros);

  tape.putInt(OP_SIGMOID);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SIGMOID_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${t.id}_grad');
    },
    opName: 'sigmoid_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for applying the tanh function on every element of vector [t] to [tape].
GPUTensor<Vector> vectorTanhGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  int N = v.shape[0];
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(N, 0.0));

  tape.putInt(OP_TANH);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(out.id);
      bTape.putString(v.id);
    },
    opName: 'tanh_vectorGPU',
    cost: N,
  );

  return out;
}
/// Appends commands for applying the tanh function on every element of matrix [m] to [tape].
GPUTensor<Matrix> tanhMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_TANH);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'tanh_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}
/// Appends commands for applying the tanh function on every element of Tensor3D [t] to [tape].
GPUTensor<Tensor3D> tanh3DGPU(GPUTensor<Tensor3D> t, CommandBuffer tape) {
  int depth = t.shape[0];
  int height = t.shape[1];
  int width = t.shape[2];

  List<List<List<double>>> zeros = <List<List<double>>>[];
  for (int c = 0; c < depth; c = c + 1) {
    List<List<double>> channel = <List<double>>[];
    for (int h = 0; h < height; h = h + 1) {
      List<double> row = <double>[];
      for (int w = 0; w < width; w = w + 1) {
        row.add(0.0);
      }
      channel.add(row);
    }
    zeros.add(channel);
  }

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>(zeros);

  tape.putInt(OP_TANH);
  tape.putString(t.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer bTape) {
      bTape.putInt(OP_TANH_BACKWARD);
      bTape.putString(out.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${t.id}_grad');
    },
    opName: 'tanh_3dGPU',
    cost: depth * height * width,
  );

  return out;
}

/// Appends commands for applying the gelu activation function on every element of vector [v] to [tape].
GPUTensor<Vector> geluGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  int n = v.shape[0];

  List<double> zeros = <double>[];
  for (int i = 0; i < n; i = i + 1) {
    zeros.add(0.0);
  }

  GPUTensor<Vector> out = GPUTensor<Vector>(zeros);

  tape.putInt(OP_GELU_FORWARD);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_GELU_BACKWARD);
      bTape.putString(v.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'geluGPU',
    cost: n,
  );

  return out;
}
/// Appends commands for applying the gelu activation function on every element of matrix [m] to [tape].
GPUTensor<Matrix> geluMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  List<List<double>> zeros = <List<double>>[];
  for (int i = 0; i < numRows; i = i + 1) {
    List<double> row = <double>[];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(0.0);
    }
    zeros.add(row);
  }

  GPUTensor<Matrix> out = GPUTensor<Matrix>(zeros);

  tape.putInt(OP_GELU_FORWARD);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_GELU_BACKWARD);
      bTape.putString(m.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'gelu_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Appends commands for applying the softmax function on every element of matrix [m] to [tape].
GPUTensor<Matrix> softmaxMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_SOFTMAX_FORWARD);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SOFTMAX_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(out.id);
      bTape.putString('${m.id}_grad');
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
GPUTensor<Scalar> binaryCrossEntropyGPU<T>(GPUTensor<T> prediction, GPUTensor<T> target, CommandBuffer tape) {
  // Loss functions usually reduce the result to a single scalar
  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_BCE_LOSS_FORWARD);
  tape.putString(prediction.id);
  tape.putString(target.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [prediction, target],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BCE_LOSS_BACKWARD);
      // CORRECT ORDER to match C++ bindings
      bTape.putString('${out.id}_grad');         // 1. Incoming grad (1.0)
      bTape.putString(prediction.id);            // 2. Prediction
      bTape.putString(target.id);                // 3. Target
      bTape.putString('${prediction.id}_grad');  // 4. Destination for the gradient
    },
    opName: 'binaryCrossEntropyGPU',
    cost: 1,
  );

  return out;
}
/// Appends commands for calculating the mean square error over between the vectors [predictions] and [targets] to [tape].
GPUTensor<Scalar> mseGPU(GPUTensor<Vector> predictions, GPUTensor<Vector> targets, CommandBuffer tape) {
  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_MSE_LOSS_FORWARD);
  tape.putString(predictions.id);
  tape.putString(targets.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [predictions, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      // CORRECT ORDER:
      bTape.putString('${out.id}_grad');         // 1. gradOut
      bTape.putString(predictions.id);           // 2. namePred
      bTape.putString(targets.id);               // 3. nameTarget
      bTape.putString('${predictions.id}_grad');  // 4. gradIn
    },
    opName: 'mseGPU',
    cost: 3 * predictions.shape[0],
  );

  return out;
}
/// Appends commands for calculating the mean square error over between the matrices [predictions] and [targets] to [tape].
GPUTensor<Scalar> mseMatrixGPU(GPUTensor<Matrix> predictions, GPUTensor<Matrix> targets, CommandBuffer tape) {
  int numRows = predictions.shape[0];
  int numCols = predictions.shape[1];

  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_MSE_LOSS_FORWARD);
  tape.putString(predictions.id);
  tape.putString(targets.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [predictions, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MSE_LOSS_BACKWARD);
      bTape.putString('${out.id}_grad');          // 1. gradOut (Der Skalar 1.0)
      bTape.putString(predictions.id);            // 2. Prediction
      bTape.putString(targets.id);                // 3. Target
      bTape.putString('${predictions.id}_grad');  // 4. gradIn (Ziel für das Update)
    },
    opName: 'mse_matrixGPU',
    cost: 3 * numRows * numCols,
  );

  return out;
}

/// /////////////////////////////////
/// Optimizers    (500-599)       ///
/// /////////////////////////////////

/// Performs an in place update of the provided [data] values via standard gradient descent, learning rate [lr] and its assigned gradient.
/// The operation is appended to [tape] for execution.
void sgdUpdateGPU(GPUTensor<dynamic> data, double lr, CommandBuffer tape) {
  tape.putInt(OP_SGD_UPDATE);
  tape.putString(data.id);
  tape.putString('${data.id}_grad'); // Automatically grab the implicit VRAM grad pointer
  tape.putFloat(lr);
}
/// Performs an in place update of the provided [data] values via the adam function and its assigned gradient.
/// The first moment [m] and second moment [v] as well as [beta1], [beta2], [eps],[weightDecay], and [steps] need to be provided.
/// The operation is appended to [tape] for execution.
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
  tape.putString('${data.id}_grad');
  tape.putString(m.id);
  tape.putString(v.id);
  tape.putFloat(lr);
  tape.putFloat(beta1);
  tape.putFloat(beta2);
  tape.putFloat(eps);
  tape.putInt(step);
  tape.putFloat(weightDecay);
}

/// Performs an in place update of the provided [data] values by clipping them in between the intervall [(-clipValue)]-[(clipValue)].
/// The operation is appended to [tape] for execution.
void clipGradValueGPU(GPUTensor<dynamic> tensor, double clipValue, CommandBuffer tape) {
  tape.putInt(OP_CLIP_GRAD_VALUE);
  // Usually applied directly to the gradients before the optimizer step
  tape.putString('${tensor.id}_grad');
  tape.putFloat(clipValue);
}

/// /////////////////////////////////
/// Reductions    (600-699)       ///
/// /////////////////////////////////

/// Appends commands for summing all elements of vector [v] to [tape].
GPUTensor<Scalar> sumGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  int N = v.shape[0];
  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_SUM_REDUCE);
  tape.putString(v.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [v],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SUM_REDUCE_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${v.id}_grad');
    },
    opName: 'sum_vectorGPU',
    cost: N,
  );

  return out;
}
/// Appends commands for summing all elements of matrix [m] to [tape].
GPUTensor<Scalar> sumMatrixGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Scalar> out = GPUTensor<Scalar>(0.0);

  tape.putInt(OP_SUM_REDUCE);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SUM_REDUCE_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'sum_matrixGPU',
    cost: numRows * numCols,
  );

  return out;
}

/// Acts as a high performance lookup of rows in a given matrix [weights] via their row indices [indices].
/// Can be used for fast odering of elements.
/// The operation is appended to [tape] for execution.
GPUTensor<Matrix> embeddingLookupGPU(GPUTensor<Vector> indices, GPUTensor<Matrix> weights, CommandBuffer tape) {
  int numIndices = indices.shape[0];
  int embeddingDim = weights.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([numIndices, embeddingDim]);

  tape.putInt(OP_EMBEDDING_FORWARD);
  tape.putString(indices.id);
  tape.putString(weights.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [weights],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EMBEDDING_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(indices.id);
      bTape.putString('${weights.id}_grad');
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
GPUTensor<Tensor3D> embeddingLookupBatchGPU(GPUTensor<Matrix> batchIndices, GPUTensor<Matrix> weights, CommandBuffer tape) {
  int batchSize = batchIndices.shape[0];
  int sequenceLength = batchIndices.shape[1];
  int embeddingDim = weights.shape[1];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty([batchSize, sequenceLength, embeddingDim]);

  tape.putInt(OP_EMBEDDING_FORWARD);
  tape.putString(batchIndices.id);
  tape.putString(weights.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    [weights],
        (CommandBuffer bTape) {
      bTape.putInt(OP_EMBEDDING_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(batchIndices.id);
      bTape.putString('${weights.id}_grad');
    },
    opName: 'embedding_lookup_batch_gpu',
    cost: batchSize * sequenceLength * embeddingDim,
  );

  return out;
}

/// Sums all matrix elements of [m] via their column order returning the sum of each column as a vector.
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> sumReduceColumnsGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int cols = m.shape[1];

  List<int> outShape = <int>[cols];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape);

  tape.putInt(OP_SUM_REDUCE_COLUMNS);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      // The backward pass of reducing columns is broadcasting the gradient back
      bTape.putInt(OP_BROADCAST_ADD);
      bTape.putString('${m.id}_grad');
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
    },
    opName: 'sumReduceColumnsGPU',
  );

  return out;
}
/// Sums all matrix elements of [m] via their row order returning the sum of each row as a vector.
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> sumReduceRowsGPU(GPUTensor<Matrix> m, CommandBuffer tape) {
  int rows = m.shape[0];

  List<int> outShape = <int>[rows];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape);

  tape.putInt(OP_SUM_REDUCE_ROWS);
  tape.putString(m.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BROADCAST_ADD);
      bTape.putString('${m.id}_grad');
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
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
GPUTensor<Matrix> sliceColumnGPU(GPUTensor<Matrix> input, int startCol, int endCol, CommandBuffer tape) {
  int rows = input.shape[0];
  int outCols = endCol - startCol;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([rows, outCols]);

  tape.putInt(OP_SLICE_COLUMN);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(startCol);
  tape.putInt(endCol);

  out.creator = GPUNode(
    [input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SLICE_COLUMN_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${input.id}_grad');
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
GPUTensor<Vector> selectRowGPU(GPUTensor<Matrix> m, int rowIndex, CommandBuffer tape) {
  int numCols = m.shape[1];
  // Output is a vector of length numCols
  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(numCols, 0.0));

  tape.putInt(OP_SLICE_ROW);
  tape.putString(m.id);
  tape.putString(out.id);
  tape.putInt(rowIndex);

  out.creator = GPUNode(
    [m],
        (CommandBuffer bTape) {
      bTape.putInt(OP_SLICE_ROW_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${m.id}_grad');
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
GPUTensor<Matrix> selectMatrixFrom3DGPU(
    GPUTensor<Tensor3D> t,
    int index,
    CommandBuffer tape) {

  int height = t.shape[1];
  int width = t.shape[2];

  List<int> outShape = <int>[height, width];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(outShape);

  tape.putInt(OP_SLICE_ROW);
  tape.putString(t.id);
  tape.putString(out.id);
  // FIX: inCols entfernt, um Tape-Desync zu verhindern!
  tape.putInt(index);

  out.creator = GPUNode(
    <GPUTensor>[t],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_SLICE_ROW_BACKWARD);
      backwardTape.putString('${out.id}_grad');
      backwardTape.putString('${t.id}_grad');
      // FIX: inCols entfernt!
      backwardTape.putInt(index);
    },
    opName: 'selectMatrixFrom3DGPU',
  );

  return out;
}

/// Appends two vectors [a] and [b] together in order [a-b].
/// The operation is appended to [tape] for execution.
GPUTensor<Vector> concatenateGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  int aLength = a.shape[0];
  int totalLength = aLength + b.shape[0];

  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(totalLength, 0.0));

  tape.putInt(OP_CONCATENATE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putInt(0);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONCATENATE_BACKWARD);
      bTape.putString(out.id);
      bTape.putString(a.id);
      bTape.putString(b.id);
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
GPUTensor<Matrix> concatenateMatricesByColumnGPU(List<GPUTensor<Matrix>> matrices, CommandBuffer tape) {
  GPUTensor<Matrix> result = matrices[0];

  for (int i = 1; i < matrices.length; i = i + 1) {
    GPUTensor<Matrix> a = result;
    GPUTensor<Matrix> b = matrices[i];

    int rows = a.shape[0];
    int colsA = a.shape[1];
    int colsB = b.shape[1];
    int totalCols = colsA + colsB;

    GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([rows, totalCols]);

    tape.putInt(OP_CONCATENATE);
    tape.putString(a.id);
    tape.putString(b.id);
    tape.putString(out.id);
    tape.putInt(1);

    out.creator = GPUNode(
      [a, b],
          (CommandBuffer bTape) {
        bTape.putInt(OP_CONCATENATE_BACKWARD);
        bTape.putString('${out.id}_grad');
        bTape.putString('${a.id}_grad');
        bTape.putString('${b.id}_grad');
        bTape.putInt(1);
        bTape.putInt(colsA);
      },
      opName: 'concat_matrix_colGPU',
      cost: 0,
    );

    result = out;
  }

  return result;
}
/// Appends two Tensor3Ds [a] and [b] together in order [a-b].
/// The operation is appended to [tape] for execution.
GPUTensor<Tensor3D> concatenate3DGPU(GPUTensor<Tensor3D> a, GPUTensor<Tensor3D> b, CommandBuffer tape) {
  int aDepth = a.shape[0];
  int bDepth = b.shape[0];
  int totalDepth = aDepth + bDepth;
  int height = a.shape[1];
  int width = a.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty([totalDepth, height, width]);

  tape.putInt(OP_CONCATENATE);
  tape.putString(a.id);
  tape.putString(b.id);
  tape.putString(out.id);
  tape.putInt(0);

  out.creator = GPUNode(
    [a, b],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONCATENATE_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${a.id}_grad');
      bTape.putString('${b.id}_grad');
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
GPUTensor<Tensor3D> stackMatricesGPU(List<GPUTensor<Matrix>> matrices, CommandBuffer tape) {
  int count = matrices.length;
  int rows = matrices[0].shape[0];
  int cols = matrices[0].shape[1];

  List<int> shape = <int>[count, rows, cols];
  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(shape);

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
      bTape.putString('${out.id}_grad');
      bTape.putInt(count);
      for (int i = 0; i < count; i = i + 1) {
        bTape.putString('${matrices[i].id}_grad');
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
GPUTensor<Matrix> scatterHeadsGPU(List<GPUTensor<Matrix>> heads, int dModel, CommandBuffer tape) {
  int seqLen = heads[0].shape[0];
  int dHead = heads[0].shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([seqLen, dModel]);

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
    [...heads],
        (CommandBuffer bTape) {
      for (int i = 0; i < heads.length; i = i + 1) {
        int startCol = i * dHead;
        int endCol = startCol + dHead;

        bTape.putInt(OP_SLICE_COLUMN);
        bTape.putString('${out.id}_grad');
        bTape.putString('${heads[i].id}_grad');
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
GPUTensor<Matrix> padMatrixGPU(GPUTensor<Matrix> input, int padSize, CommandBuffer tape) {
  int inHeight = input.shape[0];
  int inWidth = input.shape[1];
  int outHeight = inHeight + 2 * padSize;
  int outWidth = inWidth + 2 * padSize;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty([outHeight, outWidth]);

  tape.putInt(OP_PAD2D);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(padSize);
  tape.putInt(padSize);
  tape.putInt(padSize);
  tape.putInt(padSize);

  out.creator = GPUNode(
    [input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_PAD2D_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString('${input.id}_grad');
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
    int kH, int kW,
    CommandBuffer tape, {String padding = 'valid', int strideH = 1, int strideW = 1}) {

  int inChannels = input.shape.length == 2 ? 1 : input.shape[0];
  int inHeight = input.shape.length == 2 ? input.shape[0] : input.shape[1];
  int inWidth = input.shape.length == 2 ? input.shape[1] : input.shape[2];

  int outChannels = weight.shape[0];

  int padT = 0;
  int padL = 0;
  // Calculate new output dimensions based on the stride jump
  int outHeight = (inHeight - kH) ~/ strideH + 1;
  int outWidth = (inWidth - kW) ~/ strideW + 1;

  if (padding == 'same') {
    padT = (kH - 1) ~/ 2;
    padL = (kW - 1) ~/ 2;
    outHeight = (inHeight + 2 * padT - kH) ~/ strideH + 1;
    outWidth = (inWidth + 2 * padL - kW) ~/ strideW + 1;
  }

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[outChannels, outHeight, outWidth]);

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
  tape.putInt(strideH); // ⚡ Added to tape
  tape.putInt(strideW); // ⚡ Added to tape

  int cost = outHeight * outWidth * outChannels * inChannels * kH * kW * 2;

  out.creator = GPUNode(
    <GPUTensor>[input, weight, bias],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONV2D_MULTI_BACKWARD_INPUT);
      bTape.putString('${out.id}_grad');
      bTape.putString(weight.id);
      bTape.putString('${input.id}_grad');
      bTape.putInt(inChannels);
      bTape.putInt(outChannels);
      bTape.putInt(kH);
      bTape.putInt(kW);
      bTape.putInt(padT);
      bTape.putInt(padL);
      bTape.putInt(strideH); // ⚡ Added to tape
      bTape.putInt(strideW); // ⚡ Added to tape

      bTape.putInt(OP_CONV2D_MULTI_BACKWARD_WEIGHT);
      bTape.putString(input.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${weight.id}_grad');
      bTape.putString('${bias.id}_grad');
      bTape.putInt(inChannels);
      bTape.putInt(outChannels);
      bTape.putInt(kH);
      bTape.putInt(kW);
      bTape.putInt(padT);
      bTape.putInt(padL);
      bTape.putInt(strideH); // ⚡ Added to tape
      bTape.putInt(strideW); // ⚡ Added to tape
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
GPUTensor<Matrix> conv2dSimpleGPU(
    GPUTensor<Matrix> input,
    GPUTensor<Matrix> kernel,
    CommandBuffer tape) {

  int inH = input.shape[0];
  int inW = input.shape[1];
  int kH = kernel.shape[0];
  int kW = kernel.shape[1];

  int outH = inH - kH + 1;
  int outW = inW - kW + 1;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[outH, outW]);

  tape.putInt(OP_CONV2D_FORWARD);
  tape.putString(input.id);
  tape.putString(kernel.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[input, kernel],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CONV2D_BACKWARD_INPUT);
      bTape.putString('${out.id}_grad');
      bTape.putString(kernel.id);
      bTape.putString('${input.id}_grad');

      bTape.putInt(OP_CONV2D_BACKWARD_KERNEL);
      bTape.putString(input.id);
      bTape.putString('${out.id}_grad');
      bTape.putString('${kernel.id}_grad');
    },
    opName: 'conv2dSimpleGPU',
    cost: outH * outW * kH * kW,
  );

  return out;
}

/// Extracts sliding local pathces fromm an image Tensor and flattens them into columns of a matrix.
/// Kernel dimensions [kH] and [kW] need to be set.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> im2colGPU(GPUTensor<dynamic> input, int kH, int kW, CommandBuffer tape) {
  int inChannels = input.shape.length == 2 ? 1 : input.shape[0];
  int inH = input.shape.length == 2 ? input.shape[0] : input.shape[1];
  int inW = input.shape.length == 2 ? input.shape[1] : input.shape[2];

  int outH = inH - kH + 1;
  int outW = inW - kW + 1;

  int rows = inChannels * kH * kW;
  int cols = outH * outW;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[rows, cols]);

  tape.putInt(OP_IM2COL);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(kH);
  tape.putInt(kW);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_COL2IM);
      bTape.putString('${out.id}_grad');
      bTape.putString('${input.id}_grad');
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
GPUTensor<Vector> maxPool1dGPU(GPUTensor<Vector> input, int poolSize, int stride, CommandBuffer tape) {
  int inputSize = input.shape[0];
  int outputSize = (inputSize - poolSize) ~/ stride + 1;

  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(outputSize, 0.0));
  GPUTensor<Vector> indices = GPUTensor<Vector>(List<double>.filled(outputSize, 0.0));

  tape.putInt(OP_MAX_POOL_1D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(indices.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    [input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MAX_POOL_1D_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(indices.id);
      bTape.putString('${input.id}_grad');
    },
    opName: 'maxPool1dGPU',
    cost: inputSize,
  );

  return out;
}
/// Slides a pool of size [poolSize]x[poolSize] over matrix [input]. The resulting matrix consists of the maximum elements in each window.
/// The value of [stride] tells how big the step size of the sliding window is in x and y direction.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> maxPool2dGPU(GPUTensor<Matrix> input, int poolSize, int stride, CommandBuffer tape) {
  int inputHeight = input.shape[0];
  int inputWidth = input.shape[1];
  int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
  int outputWidth = (inputWidth - poolSize) ~/ stride + 1;

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[outputHeight, outputWidth]);
  GPUTensor<Matrix> indices = GPUTensor<Matrix>.empty(<int>[outputHeight, outputWidth]);

  tape.putInt(OP_MAX_POOL_2D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(indices.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    [input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_MAX_POOL_2D_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(indices.id);
      bTape.putString('${input.id}_grad');
    },
    opName: 'maxPool2dGPU',
    cost: inputHeight * inputWidth,
  );

  return out;
}

/// Slides a pool of size [poolSize]x[poolSize] over matrix [input]. The resulting matrix consists of the average of the elements in each window.
/// The value of [stride] tells how big the step size of the sliding window is.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> avgPool2dGPU(
    GPUTensor<Matrix> input,
    int poolSize,
    int stride,
    CommandBuffer tape) {

  int inHeight = input.shape[0];
  int inWidth = input.shape[1];

  int outHeight = (inHeight - poolSize) ~/ stride + 1;
  int outWidth = (inWidth - poolSize) ~/ stride + 1;

  List<int> outShape = <int>[outHeight, outWidth];
  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(outShape);

  tape.putInt(OP_AVG_POOL_2D_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putInt(poolSize);
  tape.putInt(stride);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_AVG_POOL_2D_BACKWARD);
      backwardTape.putString('${out.id}_grad');
      backwardTape.putString('${input.id}_grad');
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
    GPUTensor<Matrix> input,
    CommandBuffer tape) {

  //int sequenceLength = input.shape[0];
  int dModel = input.shape[1];

  List<int> outShape = <int>[dModel];
  GPUTensor<Vector> out = GPUTensor<Vector>.empty(outShape);

  tape.putInt(OP_GLOBAL_AVG_POOL_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer backwardTape) {
      backwardTape.putInt(OP_GLOBAL_AVG_POOL_BACKWARD);
      backwardTape.putString('${out.id}_grad');
      backwardTape.putString('${input.id}_grad');
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
    ) {
  int numFeatures = input.shape[0];

  GPUTensor<Vector> out = GPUTensor<Vector>(List<double>.filled(numFeatures, 0.0));
  GPUTensor<Vector> savedMean = GPUTensor<Vector>(List<double>.filled(numFeatures, 0.0));
  GPUTensor<Vector> savedInvVar = GPUTensor<Vector>(List<double>.filled(numFeatures, 0.0));

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
  tape.putBool(isTraining);

  out.creator = GPUNode(
    [input, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BATCH_NORM_1D_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(input.id);
      bTape.putString(gamma.id);
      bTape.putString(savedMean.id);
      bTape.putString(savedInvVar.id);
      bTape.putString('${input.id}_grad');
      bTape.putString('${gamma.id}_grad');
      bTape.putString('${beta.id}_grad');
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
    ) {
  int numChannels = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[numChannels, height, width]);
  GPUTensor<Vector> savedMean = GPUTensor<Vector>(List<double>.filled(numChannels, 0.0));
  GPUTensor<Vector> savedInvVar = GPUTensor<Vector>(List<double>.filled(numChannels, 0.0));

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
  tape.putBool(isTraining);

  out.creator = GPUNode(
    [input, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_BATCH_NORM_2D_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(input.id);
      bTape.putString(gamma.id);
      bTape.putString(savedMean.id);
      bTape.putString(savedInvVar.id);
      bTape.putString('${input.id}_grad');
      bTape.putString('${gamma.id}_grad');
      bTape.putString('${beta.id}_grad');
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
    ) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);

  tape.putInt(OP_LAYER_NORM_FORWARD);
  tape.putString(m.id);
  tape.putString(gamma.id);
  tape.putString(beta.id);
  tape.putString(out.id);
  tape.putString(meanCache.id);
  tape.putString(rstdCache.id);
  tape.putFloat(epsilon);

  out.creator = GPUNode(
    [m, gamma, beta],
        (CommandBuffer bTape) {
      bTape.putInt(OP_LAYER_NORM_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(m.id);
      bTape.putString(gamma.id);
      bTape.putString(meanCache.id);
      bTape.putString(rstdCache.id);
      bTape.putString('${m.id}_grad');
      bTape.putString('${gamma.id}_grad');
      bTape.putString('${beta.id}_grad');
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
GPUTensor<T> dropoutGPU<T>(GPUTensor<T> input, double rate, CommandBuffer tape) {
  int seed = Random().nextInt(1000000);

  GPUTensor<T> out = GPUTensor<T>.empty(input.shape);
  GPUTensor<T> mask = GPUTensor<T>.empty(input.shape);

  tape.putInt(OP_DROPOUT_FORWARD);
  tape.putString(input.id);
  tape.putString(out.id);
  tape.putString(mask.id);
  tape.putFloat(rate);
  tape.putInt(seed);

  int elementCount = 1;
  for (int i = 0; i < input.shape.length; i = i + 1) {
    elementCount = elementCount * input.shape[i];
  }

  out.creator = GPUNode(
    <GPUTensor>[input],
        (CommandBuffer bTape) {
      bTape.putInt(OP_DROPOUT_BACKWARD);
      bTape.putString('${out.id}_grad');
      bTape.putString(mask.id);
      bTape.putString('${input.id}_grad');
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
GPUTensor<Matrix> buildMarkovTableGPU(
    GPUTensor<Vector> sequence,
    int order,
    int numStates,
    CommandBuffer tape) {

  int numHistories = math.pow(numStates, order).toInt();

  // 1. Allocate VRAM for count and probability tables
  GPUTensor<Matrix> countTable = GPUTensor<Matrix>.empty(<int>[numHistories, numStates]);
  GPUTensor<Matrix> probTable = GPUTensor<Matrix>.empty(<int>[numHistories, numStates]);

  // 2. Zero out the count table (OP_FILL)
  tape.putInt(OP_FILL);
  tape.putString(countTable.id);
  tape.putFloat(0.0);

  // 3. Count Transitions
  tape.putInt(OP_MARKOV_COUNT);
  tape.putString(sequence.id);
  tape.putString(countTable.id);
  tape.putInt(order);
  tape.putInt(numStates);

  // 4. Normalize Counts to Probabilities
  tape.putInt(OP_MARKOV_NORMALIZE);
  tape.putString(countTable.id);
  tape.putString(probTable.id);
  tape.putInt(numHistories);
  tape.putInt(numStates);

  probTable.creator = GPUNode(
      <GPUTensor>[sequence],
          (CommandBuffer bTape) {}, // Empty backward pass
      opName: 'buildMarkovTableGPU'
  );

  return probTable;
}

/// Predicts the probability distribution of the next states for a batch of histories.
/// Maps each sequence in the [historyBatch] to its corresponding row in the provided [probTable]
/// and outputs a matrix of shape `[batchSize, numStates]` containing the next-state probabilities.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> markovPredictGPU(
    GPUTensor<Matrix> historyBatch, // Shape: [batch_size, order]
    GPUTensor<Matrix> probTable,    // Shape: [num_histories, num_states]
    int numStates,
    CommandBuffer tape) {

  int batchSize = historyBatch.shape[0];
  int order = historyBatch.shape[1];

  GPUTensor<Matrix> outProbs = GPUTensor<Matrix>.empty(<int>[batchSize, numStates]);

  tape.putInt(OP_MARKOV_PREDICT);
  tape.putString(historyBatch.id);
  tape.putString(probTable.id);
  tape.putString(outProbs.id);
  tape.putInt(order);
  tape.putInt(numStates);

  outProbs.creator = GPUNode(
      <GPUTensor>[historyBatch, probTable],
          (CommandBuffer bTape) {}, // Empty backward pass
      opName: 'markovPredictGPU'
  );

  return outProbs;
}


/// /////////////////////////////////
/// Fused Kernels (1000+)         ///
/// /////////////////////////////////

/// Executes a fused matrix multiplication, bias addition, and ReLU activation sequence.
/// Multiplies matrix [x] by weight matrix [w], broadcasts and adds the bias vector [b],
/// and applies a ReLU activation in a single optimized pass. Returns the activated output matrix
/// and pushes the pre-activation tensor to the [intermediates] list to manage VRAM lifecycles.
/// The operations are appended to [tape] for execution.
GPUTensor<Matrix> matMulBiasReluGPU(
    GPUTensor<Matrix> x,
    GPUTensor<Matrix> w,
    GPUTensor<Vector> b,
    CommandBuffer tape,
    List<GPUTensor> intermediates,
    ) {
  int M = x.shape[0];
  int K = x.shape[1];
  int N = w.shape[1];

  // Instantly reserve VRAM without building 33MB of Dart Lists!
  List<int> outShape = <int>[M, N];
  GPUTensor<Matrix> reluOut = GPUTensor<Matrix>.empty(outShape);
  GPUTensor<Matrix> preReluOut = GPUTensor<Matrix>.empty(outShape);

  // Add to trash so it doesn't leak VRAM!
  intermediates.add(preReluOut);

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
      bTape.putString('${reluOut.id}_grad');
      bTape.putString('${preReluOut.id}_grad');

      bTape.putInt(OP_MATMUL);
      bTape.putString(x.id);
      bTape.putString('${preReluOut.id}_grad');
      bTape.putString('${w.id}_grad');
      bTape.putBool(true);
      bTape.putBool(false);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true); // Tensor Cores ON

      bTape.putInt(OP_MATMUL);
      bTape.putString('${preReluOut.id}_grad');
      bTape.putString(w.id);
      bTape.putString('${x.id}_grad');
      bTape.putBool(false);
      bTape.putBool(true);
      bTape.putFloat(1.0);
      bTape.putFloat(1.0);
      bTape.putBool(true); // Tensor Cores ON

      bTape.putInt(OP_SUM_REDUCE_COLUMNS);
      bTape.putString('${preReluOut.id}_grad');
      bTape.putString('${b.id}_grad');
    },
    opName: 'matMulBiasReluGPU',
    cost: cost,
  );

  return reluOut;
}


/// Calculates the dot product between vector [a] and [b].
/// The operations are appended to [tape] for execution.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> dotProductGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  GPUTensor<Vector> multiplied = multiplyGPU<Vector>(a, b, tape);
  GPUTensor<Scalar> out = sumGPU(multiplied, tape);
  return out;
}

/// Computes the L2 norm (Euclidean length) of a vector. Squares each element of the vector, sums them together, and takes the square root of the result.
/// The operations are appended to [tape] for execution.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> l2NormGPU(GPUTensor<Vector> v, CommandBuffer tape) {
  // sqrt( sum( v^2 ) )
  GPUTensor<Vector> squared = powGPU<Vector>(v, 2.0, tape);
  GPUTensor<Scalar> sumOfSquares = sumGPU(squared, tape);

  return sqrtGPU<Scalar>(sumOfSquares, tape);
}

/// Calculates the straight-line Euclidean distance between two vectors.
/// Computes the element-wise difference between vector [a] and [b], squares the differences, sums them, and extracts the square root.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> euclideanDistanceGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  // sqrt( sum( (a - b)^2 ) )
  GPUTensor<Vector> diff = subtractGPU<Vector>(a, b, tape);
  GPUTensor<Vector> squaredDiff = powGPU<Vector>(diff, 2.0, tape);
  GPUTensor<Scalar> sumOfSquares = sumGPU(squaredDiff, tape);

  return sqrtGPU<Scalar>(sumOfSquares, tape);
}

/// Computes the cosine similarity between two vectors, measuring their directional alignment.
/// Calculates the dot product of [a] and [b] and divides it by the product of their individual L2 norms.
/// Returns a scalar between -1.0 (opposite) and 1.0 (identical direction).
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> cosineSimilarityGPU(GPUTensor<Vector> a, GPUTensor<Vector> b, CommandBuffer tape) {
  GPUTensor<Scalar> dot = dotProductGPU(a, b, tape);

  GPUTensor<Scalar> normA = l2NormGPU(a, tape);
  GPUTensor<Scalar> normB = l2NormGPU(b, tape);
  GPUTensor<Scalar> denominator = multiplyGPU<Scalar>(normA, normB, tape);

  return divideGPU<Scalar>(dot, denominator, tape);
}

/// Calculates the Mean Absolute Error (MAE) loss between predictions and targets.
/// Computes the absolute differences between the [preds] and [targets] vectors, sums the errors, and divides by the total number of elements to yield a single average error scalar.
/// The operations are appended to [tape] for execution.
GPUTensor<Scalar> maeLossGPU(GPUTensor<Vector> preds, GPUTensor<Vector> targets, CommandBuffer tape) {
  GPUTensor<Vector> diff = subtractGPU<Vector>(preds, targets, tape);
  GPUTensor<Vector> absoluteDiff = absGPU<Vector>(diff, tape);
  GPUTensor<Scalar> totalError = sumGPU(absoluteDiff, tape);

  GPUTensor<Scalar> nScalar = GPUTensor<Scalar>(preds.shape[0].toDouble());

  return divideGPU<Scalar>(totalError, nScalar, tape);
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
    CommandBuffer tape) {

  int numRows = input.shape[0];
  int numCols = input.shape[1];

  GPUTensor<Matrix> out = GPUTensor<Matrix>.empty(<int>[numRows, numCols]);
  GPUTensor<Vector> rstdCache = GPUTensor<Vector>.empty(<int>[numRows]);

  String weightId = '';
  String weightGradId = '';

  if (weight != null) {
    weightId = weight.id;
    weightGradId = '${weight.id}_grad';
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
      bTape.putString('${out.id}_grad');
      bTape.putString(input.id);
      bTape.putString(weightId);
      bTape.putString(rstdCache.id);
      bTape.putString('${input.id}_grad');
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
GPUTensor<Tensor3D> causalMaskGPU(
    GPUTensor<Tensor3D> input,
    int batchSize,
    int numHeads,
    int seqLen,
    CommandBuffer tape) {

  int depth = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[depth, height, width]);

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
      bTape.putString('${out.id}_grad');
      bTape.putString('${input.id}_grad');
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
    CommandBuffer tape) {

  int depth = input.shape[0];
  int height = input.shape[1];
  int width = input.shape[2];

  GPUTensor<Tensor3D> out = GPUTensor<Tensor3D>.empty(<int>[depth, height, width]);

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
      bTape.putString('${out.id}_grad');
      bTape.putString(cosTable.id);
      bTape.putString(sinTable.id);
      bTape.putString('${input.id}_grad');
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
GPUTensor<Vector> crossEntropyLossGPU(
    GPUTensor<Matrix> logits,
    GPUTensor<Vector> targets,
    CommandBuffer tape) {

  int batchSize = logits.shape[0];
  int vocabSize = logits.shape[1];

  GPUTensor<Vector> outLoss = GPUTensor<Vector>.empty(<int>[batchSize]);

  tape.putInt(OP_CROSS_ENTROPY_FORWARD);
  tape.putString(logits.id);
  tape.putString(targets.id);
  tape.putString(outLoss.id);

  outLoss.creator = GPUNode(
    <GPUTensor>[logits, targets],
        (CommandBuffer bTape) {
      bTape.putInt(OP_CROSS_ENTROPY_BACKWARD);
      bTape.putString('${outLoss.id}_grad');
      bTape.putString(logits.id);
      bTape.putString(targets.id);
      bTape.putString('${logits.id}_grad');
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
    GPUTensor<Matrix> logits,
    CommandBuffer tape) {

  int batchSize = logits.shape[0];
  int vocabSize = logits.shape[1];

  GPUTensor<Vector> outIndices = GPUTensor<Vector>.empty(<int>[batchSize]);

  tape.putInt(OP_ARGMAX_FORWARD);
  tape.putString(logits.id);
  tape.putString(outIndices.id);

  outIndices.creator = GPUNode(
    <GPUTensor>[logits],
        (CommandBuffer bTape) {
      // Inferenz-Operator, Backpropagation wird hier nicht benötigt
    },
    opName: 'argmaxGPU',
    cost: batchSize * vocabSize,
  );

  return outIndices;
}