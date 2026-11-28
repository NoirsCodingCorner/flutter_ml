import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late AllocBuffer setupAlloc;

  late GPUTensor<Scalar> scalarA;
  late GPUTensor<Scalar> scalarB;
  late GPUTensor<Scalar> scalarC;

  late GPUTensor<Vector> vectorA;
  late GPUTensor<Vector> vectorB;
  late GPUTensor<Vector> vectorC;
  late GPUTensor<Vector> vectorD;

  late GPUTensor<Matrix> matrixA;
  late GPUTensor<Matrix> matrixB;
  late GPUTensor<Matrix> matrixC;

  late GPUTensor<Tensor3D> tensor3DA;
  late GPUTensor<Tensor3D> tensor3DB;

  void testScalarGPU(GPUTensor<Scalar> actual, CommandBuffer buffer,
      AllocBuffer allocTape, double wanted,
      {double tolerance = 1e-5}) {
    allocTape.execute();
    GPUEngine.run(buffer.bytes());
    actual.dataToCpu();
    expect(actual.value, closeTo(wanted, tolerance));
    allocTape.freeAll();
  }

  void testVectorGPU(GPUTensor<Vector> actual, CommandBuffer buffer,
      AllocBuffer allocTape, List<double> wanted,
      {double tolerance = 1e-5}) {
    allocTape.execute();
    GPUEngine.run(buffer.bytes());
    actual.dataToCpu();
    for (int i = 0; i < actual.value.length; i = i + 1) {
      expect(actual.value[i], closeTo(wanted[i], tolerance));
    }
    allocTape.freeAll();
  }

  void testMatrixGPU(GPUTensor<Matrix> actual, CommandBuffer buffer,
      AllocBuffer allocTape, List<List<double>> wanted,
      {double tolerance = 1e-5}) {
    allocTape.execute();
    GPUEngine.run(buffer.bytes());
    actual.dataToCpu();
    for (int i = 0; i < actual.value.length; i = i + 1) {
      for (int j = 0; j < actual.value[i].length; j = j + 1) {
        expect(actual.value[i][j], closeTo(wanted[i][j], tolerance));
      }
    }
    allocTape.freeAll();
  }

  void testTensor3DGPU(GPUTensor<Tensor3D> actual, CommandBuffer buffer,
      AllocBuffer allocTape, List<List<List<double>>> wanted,
      {double tolerance = 1e-5}) {
    allocTape.execute();
    GPUEngine.run(buffer.bytes());
    actual.dataToCpu();
    for (int i = 0; i < actual.value.length; i = i + 1) {
      for (int j = 0; j < actual.value[i].length; j = j + 1) {
        for (int k = 0; k < actual.value[i][j].length; k = k + 1) {
          expect(actual.value[i][j][k], closeTo(wanted[i][j][k], tolerance));
        }
      }
    }
    allocTape.freeAll();
  }

  setUpAll(() async {
    await GPUEngine.initialize(target: Target.auto, debug: true);
    setupAlloc = AllocBuffer();

    scalarA = GPUTensor<Scalar>(-1.5, setupAlloc);
    scalarB = GPUTensor<Scalar>(0.0, setupAlloc);
    scalarC = GPUTensor<Scalar>(2.5, setupAlloc);
    vectorA = GPUTensor<Vector>(<double>[-1.0, 0.0, 2.0], setupAlloc);
    vectorB = GPUTensor<Vector>(<double>[0.5, -1.5, 3.0], setupAlloc);
    vectorC = GPUTensor<Vector>(<double>[4.0, -2.0, 0.0], setupAlloc);
    vectorD = GPUTensor<Vector>(<double>[-1.0, 2.0], setupAlloc);
    matrixA = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, -2.0],
      <double>[3.0, 0.0],
    ], setupAlloc);
    matrixB = GPUTensor<Matrix>(<List<double>>[
      <double>[-1.0, 0.5],
      <double>[2.0, -1.5],
    ], setupAlloc);
    matrixC = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 1.0],
      <double>[-2.0, 3.0],
    ], setupAlloc);
    tensor3DA = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
        <double>[3.0, 4.0],
      ],
    ], setupAlloc);
    tensor3DB = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[2.0, 3.0],
        <double>[4.0, 5.0],
      ],
    ], setupAlloc);

    setupAlloc.execute();
  });

  tearDownAll(() {
    setupAlloc.freeAll();
    GPUEngine.dispose();
  });

  test('addGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = addGPU(scalarA, scalarB, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, -1.5);
  });

  test('addVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        addVectorGPU(vectorA, vectorB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-0.5, -1.5, 5.0]);
  });

  test('addMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        addMatrixGPU(matrixA, matrixC, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, -1.0],
      <double>[1.0, 3.0],
    ]);
  });

  test('add3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        add3DGPU(tensor3DA, tensor3DB, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[3.0, 5.0],
        <double>[7.0, 9.0],
      ],
    ]);
  });

  test('subtractGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut =
        subtractGPU(scalarC, scalarA, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 4.0);
  });

  test('subtractVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        subtractVectorGPU(vectorA, vectorB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-1.5, 1.5, -1.0]);
  });

  test('subtractMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        subtractMatrixGPU(matrixA, matrixC, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, -3.0],
      <double>[5.0, -3.0],
    ]);
  });

  test('subtract3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        subtract3DGPU(tensor3DA, tensor3DB, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[-1.0, -1.0],
        <double>[-1.0, -1.0],
      ],
    ]);
  });

  test('multiplyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut =
        multiplyGPU(scalarA, scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, -3.75);
  });

  test('multiplyScalarGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut =
        multiplyScalarGPU(scalarA, scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, -3.75);
  });

  test('elementWiseMultiplyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        elementWiseMultiplyGPU(vectorA, vectorB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-0.5, 0.0, 6.0]);
  });

  test('elementWiseMultiplyMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        elementWiseMultiplyMatrixGPU(matrixA, matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-1.0, -1.0],
      <double>[6.0, 0.0],
    ]);
  });

  test('elementWiseMultiply3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        elementWiseMultiply3DGPU(tensor3DA, tensor3DB, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[2.0, 6.0],
        <double>[12.0, 20.0],
      ],
    ]);
  });

  test('divideGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = divideGPU(scalarA, scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, -0.6);
  });

  test('divideVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        divideVectorGPU(vectorA, vectorB, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[-2.0, 0.0, 0.6666666666666666]);
  });

  test('divideMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        divideMatrixGPU(matrixA, matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-1.0, -4.0],
      <double>[1.5, 0.0],
    ]);
  });

  test('divide3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        divide3DGPU(tensor3DA, tensor3DB, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[0.5, 0.6666666666666666],
        <double>[0.75, 0.8],
      ],
    ]);
  });

  test('vectorExpGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = vectorExpGPU(vectorA, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[0.367879, 1.0, 7.389056]);
  });

  test('absGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = absGPU(scalarA, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.5);
  });

  test('absGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = absGPU(vectorA, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 0.0, 2.0]);
  });

  test('absGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = absGPU(matrixA, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 2.0],
      <double>[3.0, 0.0],
    ]);
  });

  test('absGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = absGPU(tensor3DA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
        <double>[3.0, 4.0],
      ],
    ]);
  });

  test('sqrtGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = sqrtGPU(scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.581139);
  });

  test('sqrtGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> posVector =
        GPUTensor<Vector>(<double>[1.0, 4.0, 9.0], allocTape);
    GPUTensor<Vector> testOut = sqrtGPU(posVector, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 2.0, 3.0]);
  });

  test('sqrtGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> posMatrix = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 4.0],
      <double>[9.0, 16.0],
    ], allocTape);
    GPUTensor<Matrix> testOut = sqrtGPU(posMatrix, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 2.0],
      <double>[3.0, 4.0],
    ]);
  });

  test('sqrtGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = sqrtGPU(tensor3DA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, 1.414213],
        <double>[1.732050, 2.0],
      ],
    ]);
  });

  test('logGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = logGPU(scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 0.916291);
  });

  test('logGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> posVector =
        GPUTensor<Vector>(<double>[1.0, 2.0, 3.0], allocTape);
    GPUTensor<Vector> testOut = logGPU(posVector, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[0.0, 0.693147, 1.098612]);
  });

  test('logGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> posMatrix = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0],
      <double>[3.0, 4.0],
    ], allocTape);
    GPUTensor<Matrix> testOut = logGPU(posMatrix, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 0.693147],
      <double>[1.098612, 1.386294],
    ]);
  });

  test('logGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = logGPU(tensor3DA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[0.0, 0.693147],
        <double>[1.098612, 1.386294],
      ],
    ]);
  });

  test('powGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = powGPU(scalarC, 2.0, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 6.25);
  });

  test('powGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = powGPU(vectorA, 2.0, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 0.0, 4.0]);
  });

  test('powGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = powGPU(matrixA, 2.0, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 4.0],
      <double>[9.0, 0.0],
    ]);
  });

  test('powGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = powGPU(tensor3DA, 2.0, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, 4.0],
        <double>[9.0, 16.0],
      ],
    ]);
  });

  test('clampGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = clampGPU(scalarA, -0.5, 1.0, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, -0.5);
  });

  test('clampGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = clampGPU(vectorA, -0.5, 1.0, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-0.5, 0.0, 1.0]);
  });

  test('clampGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = clampGPU(matrixA, -0.5, 1.0, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, -0.5],
      <double>[1.0, 0.0],
    ]);
  });

  test('clampGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        clampGPU(tensor3DA, 2.0, 3.0, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[2.0, 2.0],
        <double>[3.0, 3.0],
      ],
    ]);
  });

  test('matMulGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = matMulGPU(matrixA, matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-5.0, 3.5],
      <double>[-3.0, 1.5],
    ]);
  });

  test('matVecMulGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        matVecMulGPU(matrixA, vectorD, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-5.0, -3.0]);
  });

  test('transposeGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = transposeGPU(matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-1.0, 2.0],
      <double>[0.5, -1.5],
    ]);
  });

  test('addMatrixAndVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        addMatrixAndVectorGPU(matrixA, vectorD, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 0.0],
      <double>[2.0, 2.0],
    ]);
  });

  test('addScalarMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        addScalarMatrixGPU(matrixA, scalarA, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-0.5, -3.5],
      <double>[1.5, -1.5],
    ]);
  });

  test('addScalarVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        addScalarVectorGPU(vectorA, 2.0, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 2.0, 4.0]);
  });

  test('addScalar3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        addScalar3DGPU(tensor3DA, 2.0, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[3.0, 4.0],
        <double>[5.0, 6.0],
      ],
    ]);
  });

  test('addBiasToFeatureMapGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> bias = GPUTensor<Matrix>(<List<double>>[
      <double>[2.0]
    ], allocTape);
    GPUTensor<Matrix> testOut =
        addBiasToFeatureMapGPU(matrixA, bias, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[3.0, 0.0],
      <double>[5.0, 2.0],
    ]);
  });

  test('addBiasToMatMulOutGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        addBiasToMatMulOutGPU(matrixA, vectorD, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 0.0],
      <double>[2.0, 2.0],
    ]);
  });

  test('broadcastAddVectorToMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        broadcastAddVectorToMatrixGPU(matrixA, vectorD, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 0.0],
      <double>[2.0, 2.0],
    ]);
  });

  test('scaleMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = scaleMatrixGPU(matrixA, 2.0, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[2.0, -4.0],
      <double>[6.0, 0.0],
    ]);
  });

  test('reluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = reluGPU(vectorB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[0.5, 0.0, 3.0]);
  });

  test('reluMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = reluMatrixGPU(matrixA, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 0.0],
      <double>[3.0, 0.0],
    ]);
  });

  test('sigmoidScalarGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut =
        sigmoidScalarGPU<Scalar>(scalarC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 0.92414);
  });

  test('sigmoidGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = sigmoidGPU(vectorC, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[0.982013, 0.119202, 0.5]);
  });

  test('sigmoidMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = sigmoidMatrixGPU(matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.268941, 0.622459],
      <double>[0.880797, 0.182425],
    ]);
  });

  test('sigmoid3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = sigmoid3DGPU(tensor3DA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[0.731058, 0.880797],
        <double>[0.952574, 0.982013],
      ],
    ]);
  });

  test('vectorTanhGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = vectorTanhGPU(vectorB, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[0.462117, -0.905148, 0.995055]);
  });

  test('tanhMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = tanhMatrixGPU(matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-0.761594, 0.462117],
      <double>[0.964027, -0.905148],
    ]);
  });

  test('tanh3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = tanh3DGPU(tensor3DA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[0.761594, 0.964027],
        <double>[0.995055, 0.999329],
      ],
    ]);
  });

  test('geluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = geluGPU(vectorA, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[-0.158655, 0.0, 1.954499],
        tolerance: 5e-4);
  });

  test('geluMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = geluMatrixGPU(matrixA, buffer, allocTape);
    testMatrixGPU(
        testOut,
        buffer,
        allocTape,
        <List<double>>[
          <double>[0.841345, -0.045500],
          <double>[2.995950, 0.0],
        ],
        tolerance: 5e-4);
  });

  test('softmaxMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = softmaxMatrixGPU(matrixB, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.182425, 0.817575],
      <double>[0.970687, 0.029312],
    ]);
  });

  test('binaryCrossEntropyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> small = GPUTensor<Scalar>(0.7, allocTape);
    GPUTensor<Scalar> big = GPUTensor<Scalar>(0.9, allocTape);
    GPUTensor<Scalar> testOut =
        binaryCrossEntropyGPU<Scalar>(small, big, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 0.44140473);
  });

  test('mseGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = mseGPU(vectorA, vectorB, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.8333333);
  });

  test('mseMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut =
        mseMatrixGPU(matrixA, matrixB, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 3.375);
  });

  test('sgdUpdateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>(<double>[1.0, 2.0], allocTape);

    sgdUpdateGPU(data, 0.1, buffer);
    allocTape.execute();
    GPUEngine.run(buffer.bytes());

    data.dataToCpu();
    expect(data.value[0], closeTo(1.0, 1e-5));
    allocTape.freeAll();
  });

  test('adamUpdateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>(<double>[1.0, 2.0], allocTape);
    GPUTensor<Vector> m = GPUTensor<Vector>(<double>[0.0, 0.0], allocTape);
    GPUTensor<Vector> v = GPUTensor<Vector>(<double>[0.0, 0.0], allocTape);

    adamUpdateGPU(data, m, v, 0.01, 0.9, 0.999, 1e-8, 1, 0.0, buffer);
    allocTape.execute();
    GPUEngine.run(buffer.bytes());

    data.dataToCpu();
    expect(data.value[0], closeTo(1.0, 1e-5));
    allocTape.freeAll();
  });

  test('clipGradValueGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>(<double>[1.0, 2.0], allocTape);

    clipGradValueGPU(data, 1.0, buffer);
    allocTape.execute();
    GPUEngine.run(buffer.bytes());

    expect(true, true);
    allocTape.freeAll();
  });

  test('sumGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = sumGPU(vectorC, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 2.0);
  });

  test('sumMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Scalar> testOut = sumMatrixGPU(matrixB, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 0.0);
  });

  test('embeddingLookupGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> indices =
        GPUTensor<Vector>(<double>[1.0, 0.0], allocTape);
    GPUTensor<Matrix> testOut =
        embeddingLookupGPU(indices, matrixA, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[3.0, 0.0],
      <double>[1.0, -2.0],
    ]);
  });

  test('embeddingLookupBatchGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> batchIndices = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 0.0],
      <double>[0.0, 1.0],
    ], allocTape);
    GPUTensor<Tensor3D> testOut =
        embeddingLookupBatchGPU(batchIndices, matrixA, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[3.0, 0.0],
        <double>[1.0, -2.0],
      ],
      <List<double>>[
        <double>[1.0, -2.0],
        <double>[3.0, 0.0],
      ],
    ]);
  });

  test('sumReduceColumnsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = sumReduceColumnsGPU(matrixB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, -1.0]);
  });

  test('sumReduceRowsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = sumReduceRowsGPU(matrixB, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[-0.5, 0.5]);
  });

  test('sliceColumnGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        sliceColumnGPU(matrixA, 0, 1, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0],
      <double>[3.0],
    ]);
  });

  test('selectRowGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut = selectRowGPU(matrixB, 1, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[2.0, -1.5]);
  });

  test('selectMatrixFrom3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut =
        selectMatrixFrom3DGPU(tensor3DA, 0, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 2.0],
      <double>[3.0, 4.0],
    ]);
  });

  test('concatenateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> testOut =
        concatenateGPU(vectorA, vectorB, buffer, allocTape);
    testVectorGPU(
        testOut, buffer, allocTape, <double>[-1.0, 0.0, 2.0, 0.5, -1.5, 3.0]);
  });

  test('concatenateMatricesByColumnGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = concatenateMatricesByColumnGPU(
        <GPUTensor<Matrix>>[matrixA, matrixB], buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, -2.0, -1.0, 0.5],
      <double>[3.0, 0.0, 2.0, -1.5],
    ]);
  });

  test('concatenate3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut =
        concatenate3DGPU(tensor3DA, tensor3DB, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
        <double>[3.0, 4.0],
      ],
      <List<double>>[
        <double>[2.0, 3.0],
        <double>[4.0, 5.0],
      ],
    ]);
  });

  test('stackMatricesGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> testOut = stackMatricesGPU(
        <GPUTensor<Matrix>>[matrixA, matrixB], buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, -2.0],
        <double>[3.0, 0.0],
      ],
      <List<double>>[
        <double>[-1.0, 0.5],
        <double>[2.0, -1.5],
      ],
    ]);
  });

  test('scatterHeadsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = scatterHeadsGPU(
        <GPUTensor<Matrix>>[matrixA, matrixB], 4, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, -2.0, -1.0, 0.5],
      <double>[3.0, 0.0, 2.0, -1.5],
    ]);
  });

  test('padMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> testOut = padMatrixGPU(matrixC, 1, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 0.0, 0.0, 0.0],
      <double>[0.0, 0.0, 1.0, 0.0],
      <double>[0.0, -2.0, 3.0, 0.0],
      <double>[0.0, 0.0, 0.0, 0.0],
    ]);
  });

  test('conv2dMultiChannelGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0, 3.0],
      <double>[4.0, 5.0, 6.0],
      <double>[7.0, 8.0, 9.0],
    ], allocTape);
    GPUTensor<Tensor3D> weight = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[1.0, 0.0],
        <double>[0.0, -1.0],
      ],
    ], allocTape);
    GPUTensor<Vector> bias = GPUTensor<Vector>(<double>[0.0], allocTape);

    GPUTensor<Tensor3D> testOut =
        conv2dMultiChannelGPU(input, weight, bias, 2, 2, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[-4.0, -4.0],
        <double>[-4.0, -4.0],
      ],
    ]);
  });

  test('conv2dSimpleGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0, 3.0],
      <double>[4.0, 5.0, 6.0],
      <double>[7.0, 8.0, 9.0],
    ], allocTape);
    GPUTensor<Matrix> kernel = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 0.0],
      <double>[0.0, -1.0],
    ], allocTape);

    GPUTensor<Matrix> testOut =
        conv2dSimpleGPU(input, kernel, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-4.0, -4.0],
      <double>[-4.0, -4.0],
    ]);
  });

  test('im2colGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0, 3.0],
      <double>[4.0, 5.0, 6.0],
      <double>[7.0, 8.0, 9.0],
    ], allocTape);

    GPUTensor<Matrix> testOut = im2colGPU(input, 2, 2, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[1.0, 2.0, 4.0, 5.0],
      <double>[2.0, 3.0, 5.0, 6.0],
      <double>[4.0, 5.0, 7.0, 8.0],
      <double>[5.0, 6.0, 8.0, 9.0],
    ]);
  });

  test('maxPool1dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> input =
        GPUTensor<Vector>(<double>[1.0, 3.0, 2.0, 5.0, 4.0], allocTape);

    GPUTensor<Vector> testOut = maxPool1dGPU(input, 2, 2, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[3.0, 5.0]);
  });

  test('maxPool2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0, 3.0, 4.0],
      <double>[5.0, 6.0, 7.0, 8.0],
      <double>[9.0, 10.0, 11.0, 12.0],
      <double>[13.0, 14.0, 15.0, 16.0],
    ], allocTape);

    GPUTensor<Matrix> testOut = maxPool2dGPU(input, 2, 2, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[6.0, 8.0],
      <double>[14.0, 16.0],
    ]);
  });

  test('avgPool2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0, 3.0, 4.0],
      <double>[5.0, 6.0, 7.0, 8.0],
      <double>[9.0, 10.0, 11.0, 12.0],
      <double>[13.0, 14.0, 15.0, 16.0],
    ], allocTape);

    GPUTensor<Matrix> testOut = avgPool2dGPU(input, 2, 2, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[3.5, 5.5],
      <double>[11.5, 13.5],
    ]);
  });

  test('globalAveragePoolingGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 2.0],
      <double>[3.0, 4.0],
      <double>[5.0, 6.0],
    ], allocTape);

    GPUTensor<Vector> testOut =
        globalAveragePoolingGPU(input, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[3.0, 4.0]);
  });

  test('batchNorm1dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> x = GPUTensor<Vector>(<double>[2.0, 4.0], allocTape);
    GPUTensor<Vector> gamma = GPUTensor<Vector>(<double>[1.0, 2.0], allocTape);
    GPUTensor<Vector> beta = GPUTensor<Vector>(<double>[0.0, 1.0], allocTape);
    GPUTensor<Vector> runningMean =
        GPUTensor<Vector>(<double>[0.0, 2.0], allocTape);
    GPUTensor<Vector> runningVar =
        GPUTensor<Vector>(<double>[4.0, 4.0], allocTape);

    GPUTensor<Vector> testOut = batchNorm1dGPU(
      x,
      gamma,
      beta,
      runningMean,
      runningVar,
      0.9,
      0.0,
      false,
      buffer,
      allocTape,
    );
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 3.0]);
  });

  test('batchNorm2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> x = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
        <double>[3.0, 4.0],
      ],
    ], allocTape);
    GPUTensor<Vector> gamma = GPUTensor<Vector>(<double>[2.0], allocTape);
    GPUTensor<Vector> beta = GPUTensor<Vector>(<double>[1.0], allocTape);
    GPUTensor<Vector> runningMean = GPUTensor<Vector>(<double>[2.5], allocTape);
    GPUTensor<Vector> runningVar = GPUTensor<Vector>(<double>[1.0], allocTape);

    GPUTensor<Tensor3D> testOut = batchNorm2dGPU(
      x,
      gamma,
      beta,
      runningMean,
      runningVar,
      0.9,
      0.0,
      false,
      buffer,
      allocTape,
    );
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[-2.0, 0.0],
        <double>[2.0, 4.0],
      ],
    ]);
  });

  test('layerNormMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> m = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 3.0],
      <double>[2.0, 4.0],
    ], allocTape);
    GPUTensor<Vector> gamma = GPUTensor<Vector>(<double>[1.0, 1.0], allocTape);
    GPUTensor<Vector> beta = GPUTensor<Vector>(<double>[0.0, 0.0], allocTape);
    GPUTensor<Vector> meanCache =
        GPUTensor<Vector>(<double>[0.0, 0.0], allocTape);
    GPUTensor<Vector> rstdCache =
        GPUTensor<Vector>(<double>[0.0, 0.0], allocTape);

    GPUTensor<Matrix> testOut = layerNormMatrixGPU(
      m,
      gamma,
      beta,
      meanCache,
      rstdCache,
      0.0,
      buffer,
      allocTape,
    );
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[-1.0, 1.0],
      <double>[-1.0, 1.0],
    ]);
  });

  test('dropoutGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> input =
        GPUTensor<Vector>(<double>[1.0, 2.0, 3.0], allocTape);

    GPUTensor<Vector> testOut = dropoutGPU(input, 0.0, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 2.0, 3.0]);
  });

  test('buildMarkovTableGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> sequence =
        GPUTensor<Vector>(<double>[0.0, 1.0, 0.0, 1.0, 1.0], allocTape);

    GPUTensor<Matrix> testOut =
        buildMarkovTableGPU(sequence, 1, 2, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 1.0],
      <double>[0.5, 0.5],
    ]);
  });

  test('markovPredictGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> historyBatch = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0],
      <double>[0.0],
    ], allocTape);
    GPUTensor<Matrix> probTable = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 1.0],
      <double>[0.5, 0.5],
    ], allocTape);

    GPUTensor<Matrix> testOut =
        markovPredictGPU(historyBatch, probTable, 2, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.5, 0.5],
      <double>[0.0, 1.0],
    ]);
  });

  test('matMulBiasReluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> x = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, -2.0],
      <double>[3.0, 0.0],
    ], allocTape);
    GPUTensor<Matrix> w = GPUTensor<Matrix>(<List<double>>[
      <double>[-1.0, 2.0],
      <double>[0.5, -1.5],
    ], allocTape);
    GPUTensor<Vector> b = GPUTensor<Vector>(<double>[1.0, -1.0], allocTape);

    GPUTensor<Matrix> testOut = matMulBiasReluGPU(x, w, b, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.0, 4.0],
      <double>[0.0, 5.0],
    ]);

    expect(testOut.subMap.containsKey('preReluOut'), true);
  });

  test('dotProductGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>(<double>[1.0, 2.0, 3.0], allocTape);
    GPUTensor<Vector> b =
        GPUTensor<Vector>(<double>[2.0, -1.0, 0.5], allocTape);

    GPUTensor<Scalar> testOut = dotProductGPU(a, b, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.5);
  });

  test('l2NormGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> v = GPUTensor<Vector>(<double>[3.0, -4.0], allocTape);

    GPUTensor<Scalar> testOut = l2NormGPU(v, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 5.0);
  });

  test('euclideanDistanceGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>(<double>[1.0, 2.0, 3.0], allocTape);
    GPUTensor<Vector> b =
        GPUTensor<Vector>(<double>[4.0, 2.0, -1.0], allocTape);

    GPUTensor<Scalar> testOut = euclideanDistanceGPU(a, b, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 5.0);
  });

  test('cosineSimilarityGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>(<double>[3.0, 4.0], allocTape);
    GPUTensor<Vector> b = GPUTensor<Vector>(<double>[6.0, 8.0], allocTape);

    GPUTensor<Scalar> testOut = cosineSimilarityGPU(a, b, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.0);
  });

  test('maeLossGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Vector> preds =
        GPUTensor<Vector>(<double>[1.0, 3.0, 5.0], allocTape);
    GPUTensor<Vector> targets =
        GPUTensor<Vector>(<double>[2.0, 1.0, 5.0], allocTape);

    GPUTensor<Scalar> testOut = maeLossGPU(preds, targets, buffer, allocTape);
    testScalarGPU(testOut, buffer, allocTape, 1.0);
  });

  test('rmsNormMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>(<List<double>>[
      <double>[3.0, 4.0],
    ], allocTape);
    GPUTensor<Vector> weight = GPUTensor<Vector>(<double>[1.0, 1.0], allocTape);

    GPUTensor<Matrix> testOut =
        rmsNormMatrixGPU(input, weight, 0.0, false, buffer, allocTape);
    testMatrixGPU(testOut, buffer, allocTape, <List<double>>[
      <double>[0.848528, 1.131371],
    ]);
  });

  test('causalMaskGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
        <double>[3.0, 4.0],
      ],
    ], allocTape);

    GPUTensor<Tensor3D> testOut =
        causalMaskGPU(input, 1, 1, 2, buffer, allocTape);
    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[1.0, -10000.0],
        <double>[3.0, 4.0],
      ],
    ]);
  });

  test('applyRopeGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>(<List<List<double>>>[
      <List<double>>[
        <double>[1.0, 2.0],
      ],
    ], allocTape);

    GPUTensor<Matrix> cosTable = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0]
    ], allocTape);
    GPUTensor<Matrix> sinTable = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0]
    ], allocTape);

    GPUTensor<Tensor3D> testOut =
        applyRopeGPU(input, cosTable, sinTable, 1, 1, 1, 2, buffer, allocTape);

    testTensor3DGPU(testOut, buffer, allocTape, <List<List<double>>>[
      <List<double>>[
        <double>[-2.0, 1.0],
      ],
    ]);
  });

  test('crossEntropyLossGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> logits = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 1.0, 0.0],
    ], allocTape);
    GPUTensor<Vector> targets = GPUTensor<Vector>(<double>[1.0], allocTape);

    GPUTensor<Vector> testOut =
        crossEntropyLossGPU(logits, targets, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[0.551444]);
  });

  test('argmaxGPU', () {
    CommandBuffer buffer = CommandBuffer();
    AllocBuffer allocTape = AllocBuffer();
    GPUTensor<Matrix> logits = GPUTensor<Matrix>(<List<double>>[
      <double>[1.0, 5.0, 2.0],
      <double>[8.0, -1.0, 3.0],
    ], allocTape);

    GPUTensor<Vector> testOut = argmaxGPU(logits, buffer, allocTape);
    testVectorGPU(testOut, buffer, allocTape, <double>[1.0, 0.0]);
  });
}
