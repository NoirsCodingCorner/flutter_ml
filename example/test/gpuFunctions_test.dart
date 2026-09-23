import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
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

  void testScalarGPU(GPUTensor<Scalar> actual, CommandBuffer buffer, double wanted,{tolerance = 1e-5}) {
    GPUEngine.run(buffer.bytes());
    actual.toCpu();
    expect(actual.value, closeTo(wanted, tolerance));
  }
  void testVectorGPU(GPUTensor<Vector> actual, CommandBuffer buffer, List<double> wanted,
      {tolerance = 1e-5}) {
    GPUEngine.run(buffer.bytes());
    actual.toCpu();
    for (int i = 0; i < actual.value.length; i++) {
      expect(actual.value[i], closeTo(wanted[i], tolerance));
    }
  }
  void testMatrixGPU(GPUTensor<Matrix> actual, CommandBuffer buffer, List<List<double>> wanted,{tolerance = 1e-5}) {
    GPUEngine.run(buffer.bytes());
    actual.toCpu();
    for (int i = 0; i < actual.value.length; i++) {
      for (int j = 0; j < actual.value[i].length; j++) {
        expect(actual.value[i][j], closeTo(wanted[i][j], tolerance));
      }
    }
  }
  void testTensor3DGPU(
      GPUTensor<Tensor3D> actual,
      CommandBuffer buffer,
      List<List<List<double>>> wanted,{tolerance = 1e-5}
      ) {
    GPUEngine.run(buffer.bytes());
    actual.toCpu();
    for (int i = 0; i < actual.value.length; i++) {
      for (int j = 0; j < actual.value[i].length; j++) {
        for (int k = 0; k < actual.value[i][j].length; k++) {
          expect(actual.value[i][j][k], closeTo(wanted[i][j][k], tolerance));
        }
      }
    }
  }

  setUpAll(() async {
    await GPUEngine.initialize(target: Target.cuda,debug: true);
    scalarA = GPUTensor<Scalar>(-1.5);
    scalarB = GPUTensor<Scalar>(0.0);
    scalarC = GPUTensor<Scalar>(2.5);
    vectorA = GPUTensor<Vector>([-1.0, 0.0, 2.0]);
    vectorB = GPUTensor<Vector>([0.5, -1.5, 3.0]);
    vectorC = GPUTensor<Vector>([4.0, -2.0, 0.0]);
    vectorD = GPUTensor<Vector>([-1.0, 2.0]);
    matrixA = GPUTensor<Matrix>([[1.0, -2.0], [3.0, 0.0],]);
    matrixB = GPUTensor<Matrix>([[-1.0, 0.5], [2.0, -1.5],]);
    matrixC = GPUTensor<Matrix>([[0.0, 1.0], [-2.0, 3.0],]);
    tensor3DA = GPUTensor<Tensor3D>([
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
    ]);
    tensor3DB = GPUTensor<Tensor3D>([
      [
        [2.0, 3.0],
        [4.0, 5.0],
      ],
    ]);
  });


  test('addGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = addGPU(scalarA, scalarB, buffer);
    testScalarGPU(testOut, buffer, -1.5);
  });
  test('addVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = addVectorGPU(vectorA, vectorB, buffer);
    testVectorGPU(testOut, buffer, [-0.5, -1.5, 5.0]);
  });
  test('addMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = addMatrixGPU(matrixA, matrixC, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, -1.0],
      [1.0, 3.0],
    ]);
  });
  test('add3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = add3DGPU(tensor3DA, tensor3DB, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [3.0, 5.0],
        [7.0, 9.0],
      ],
    ]);
  });

  test('subtractGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = subtractGPU(scalarC, scalarA, buffer);
    testScalarGPU(testOut, buffer, 4.0);
  });
  test('subtractVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = subtractVectorGPU(vectorA, vectorB, buffer);
    testVectorGPU(testOut, buffer, [-1.5, 1.5, -1.0]);
  });
  test('subtractMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = subtractMatrixGPU(matrixA, matrixC, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, -3.0],
      [5.0, -3.0],
    ]);
  });
  test('subtract3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = subtract3DGPU(tensor3DA, tensor3DB, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [-1.0, -1.0],
        [-1.0, -1.0],
      ],
    ]);
  });

  test('multiplyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = multiplyGPU(scalarA, scalarC, buffer);
    testScalarGPU(testOut, buffer, -3.75);
  });
  test('multiplyScalarGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = multiplyScalarGPU(scalarA, scalarC, buffer);
    testScalarGPU(testOut, buffer, -3.75);
  });
  test('elementWiseMultiplyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = elementWiseMultiplyGPU(vectorA, vectorB, buffer);
    testVectorGPU(testOut, buffer, [-0.5, 0.0, 6.0]);
  });
  test('elementWiseMultiplyMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = elementWiseMultiplyMatrixGPU(matrixA, matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [-1.0, -1.0],
      [6.0, 0.0],
    ]);
  });
  test('elementWiseMultiply3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = elementWiseMultiply3DGPU(tensor3DA, tensor3DB, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [2.0, 6.0],
        [12.0, 20.0],
      ],
    ]);
  });

  test('divideGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = divideGPU(scalarA, scalarC, buffer);
    testScalarGPU(testOut, buffer, -0.6);
  });
  test('divideVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = divideVectorGPU(vectorA, vectorB, buffer);
    testVectorGPU(testOut, buffer, [-2.0, 0.0, 0.6666666666666666]);
  });
  test('divideMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = divideMatrixGPU(matrixA, matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [-1.0, -4.0],
      [1.5, 0.0],
    ]);
  });
  test('divide3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = divide3DGPU(tensor3DA, tensor3DB, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [0.5, 0.6666666666666666],
        [0.75, 0.8],
      ],
    ]);
  });

  test('vectorExpGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = vectorExpGPU(vectorA, buffer);
    testVectorGPU(testOut, buffer, [0.367879, 1.0, 7.389056]);
  });

  test('absGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = absGPU(scalarA, buffer);
    testScalarGPU(testOut, buffer, 1.5);
  });
  test('absGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = absGPU(vectorA, buffer);
    testVectorGPU(testOut, buffer, [1.0, 0.0, 2.0]);
  });
  test('absGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = absGPU(matrixA, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 2.0],
      [3.0, 0.0],
    ]);
  });
  test('absGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = absGPU(tensor3DA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
    ]);
  });

  test('sqrtGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = sqrtGPU(scalarC, buffer);
    testScalarGPU(testOut, buffer, 1.581139);
  });
  test('sqrtGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> posVector = GPUTensor<Vector>([1.0, 4.0, 9.0]);
    GPUTensor<Vector> testOut = sqrtGPU(posVector, buffer);
    testVectorGPU(testOut, buffer, [1.0, 2.0, 3.0]);
  });
  test('sqrtGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> posMatrix = GPUTensor<Matrix>([
      [1.0, 4.0],
      [9.0, 16.0],
    ]);
    GPUTensor<Matrix> testOut = sqrtGPU(posMatrix, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });
  test('sqrtGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = sqrtGPU(tensor3DA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, 1.414213],
        [1.732050, 2.0],
      ],
    ]);
  });

  test('logGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = logGPU(scalarC, buffer);
    testScalarGPU(testOut, buffer, 0.916291);
  });
  test('logGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> posVector = GPUTensor<Vector>([1.0, 2.0, 3.0]);
    GPUTensor<Vector> testOut = logGPU(posVector, buffer);
    testVectorGPU(testOut, buffer, [0.0, 0.693147, 1.098612]);
  });
  test('logGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> posMatrix = GPUTensor<Matrix>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    GPUTensor<Matrix> testOut = logGPU(posMatrix, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 0.693147],
      [1.098612, 1.386294],
    ]);
  });
  test('logGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = logGPU(tensor3DA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [0.0, 0.693147],
        [1.098612, 1.386294],
      ],
    ]);
  });

  test('powGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = powGPU(scalarC, 2.0, buffer);
    testScalarGPU(testOut, buffer, 6.25);
  });


  test('powGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = powGPU(vectorA, 2.0, buffer);
    testVectorGPU(testOut, buffer, [1.0, 0.0, 4.0]);
  });
  test('powGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = powGPU(matrixA, 2.0, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 4.0],
      [9.0, 0.0],
    ]);
  });

  test('powGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = powGPU(tensor3DA, 2.0, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, 4.0],
        [9.0, 16.0],
      ],
    ]);
  });

  test('clampGPU_Scalar', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = clampGPU(scalarA, -0.5, 1.0, buffer);
    testScalarGPU(testOut, buffer, -0.5);
  });
  test('clampGPU_Vector', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = clampGPU(vectorA, -0.5, 1.0, buffer);
    testVectorGPU(testOut, buffer, [-0.5, 0.0, 1.0]);
  });
  test('clampGPU_Matrix', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = clampGPU(matrixA, -0.5, 1.0, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, -0.5],
      [1.0, 0.0],
    ]);
  });
  test('clampGPU_Tensor3D', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = clampGPU(tensor3DA, 2.0, 3.0, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [2.0, 2.0],
        [3.0, 3.0],
      ],
    ]);
  });

  test('matMulGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = matMulGPU(matrixA, matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [-5.0, 3.5],
      [-3.0, 1.5],
    ]);
  });
  test('matVecMulGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = matVecMulGPU(matrixA, vectorD, buffer);
    testVectorGPU(testOut, buffer, [-5.0, -3.0]);
  });

  test('transposeGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = transposeGPU(matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [-1.0, 2.0],
      [0.5, -1.5],
    ]);
  });

  test('addMatrixAndVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = addMatrixAndVectorGPU(matrixA, vectorD, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 0.0],
      [2.0, 2.0],
    ]);
  });

  test('addScalarMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = addScalarMatrixGPU(matrixA, scalarA, buffer);
    testMatrixGPU(testOut, buffer, [
      [-0.5, -3.5],
      [1.5, -1.5],
    ]);
  });
  test('addScalarVectorGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = addScalarVectorGPU(vectorA, 2.0, buffer);
    testVectorGPU(testOut, buffer, [1.0, 2.0, 4.0]);
  });
  test('addScalar3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = addScalar3DGPU(tensor3DA, 2.0, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [3.0, 4.0],
        [5.0, 6.0],
      ],
    ]);
  });

  test('addBiasToFeatureMapGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> bias = GPUTensor<Matrix>([[2.0]]);
    GPUTensor<Matrix> testOut = addBiasToFeatureMapGPU(matrixA, bias, buffer);
    testMatrixGPU(testOut, buffer, [
      [3.0, 0.0],
      [5.0, 2.0],
    ]);
  });
  test('addBiasToMatMulOutGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = addBiasToMatMulOutGPU(matrixA, vectorD, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 0.0],
      [2.0, 2.0],
    ]);
  });
  test('broadcastAddVectorToMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = broadcastAddVectorToMatrixGPU(matrixA, vectorD, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 0.0],
      [2.0, 2.0],
    ]);
  });
  test('scaleMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = scaleMatrixGPU(matrixA, 2.0, buffer);
    testMatrixGPU(testOut, buffer, [
      [2.0, -4.0],
      [6.0, 0.0],
    ]);
  });

  test('reluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = reluGPU(vectorB, buffer);
    testVectorGPU(testOut, buffer, [0.5, 0.0, 3.0]);
  });
  test('reluMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = reluMatrixGPU(matrixA, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 0.0],
      [3.0, 0.0],
    ]);
  });
  test('sigmoidScalarGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = sigmoidScalarGPU<Scalar>(scalarC, buffer);
    testScalarGPU(testOut, buffer, 0.92414);
  });
  test('sigmoidGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = sigmoidGPU(vectorC, buffer);
    testVectorGPU(testOut, buffer, [0.982013, 0.119202, 0.5]);
  });
  test('sigmoidMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = sigmoidMatrixGPU(matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.268941, 0.622459],
      [0.880797, 0.182425],
    ]);
  });
  test('sigmoid3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = sigmoid3DGPU(tensor3DA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [0.731058, 0.880797],
        [0.952574, 0.982013],
      ],
    ]);
  });

  test('vectorTanhGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = vectorTanhGPU(vectorB, buffer);
    testVectorGPU(testOut, buffer, [0.462117, -0.905148, 0.995055]);
  });
  test('tanhMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = tanhMatrixGPU(matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [-0.761594, 0.462117],
      [0.964027, -0.905148],
    ]);
  });
  test('tanh3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = tanh3DGPU(tensor3DA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [0.761594, 0.964027],
        [0.995055, 0.999329],
      ],
    ]);
  });

  /// This needs a leaner tolerance on WebGPU
  test('geluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = geluGPU(vectorA, buffer);
    testVectorGPU(testOut, buffer, [-0.158655, 0.0, 1.954499],tolerance: 5e-4);
  });
  test('geluMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = geluMatrixGPU(matrixA, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.841345, -0.045500],
      [2.995950, 0.0],
    ],tolerance: 5e-4);
  });
  test('softmaxMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = softmaxMatrixGPU(matrixB, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.182425, 0.817575],
      [0.970687, 0.029312],
    ]);
  });

  test('binaryCrossEntropyGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> small = GPUTensor<Scalar>(0.7);
    GPUTensor<Scalar> big = GPUTensor<Scalar>(0.9);
    GPUTensor<Scalar> testOut = binaryCrossEntropyGPU<Scalar>(small, big, buffer);
    testScalarGPU(testOut, buffer, 0.44140473);
  });
  test('mseGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = mseGPU(vectorA, vectorB, buffer);
    testScalarGPU(testOut, buffer, 1.8333333);
  });
  test('mseMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = mseMatrixGPU(matrixA, matrixB, buffer);
    testScalarGPU(testOut, buffer, 3.375);
  });

  test('sgdUpdateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>([1.0, 2.0]);

    sgdUpdateGPU(data, 0.1, buffer);
    GPUEngine.run(buffer.bytes());

    data.toCpu();
    expect(data.value[0], closeTo(1.0, 1e-5));
  });
  test('adamUpdateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>([1.0, 2.0]);
    GPUTensor<Vector> m = GPUTensor<Vector>([0.0, 0.0]);
    GPUTensor<Vector> v = GPUTensor<Vector>([0.0, 0.0]);

    adamUpdateGPU(data, m, v, 0.01, 0.9, 0.999, 1e-8, 1, 0.0, buffer);
    GPUEngine.run(buffer.bytes());

    data.toCpu();
    expect(data.value[0], closeTo(1.0, 1e-5));
  });
  test('clipGradValueGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> data = GPUTensor<Vector>([1.0, 2.0]);

    clipGradValueGPU(data, 1.0, buffer);
    GPUEngine.run(buffer.bytes());

    expect(true, true);
  });

  test('sumGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = sumGPU(vectorC, buffer);
    testScalarGPU(testOut, buffer, 2.0); // 4.0 - 2.0 + 0.0
  });
  test('sumMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Scalar> testOut = sumMatrixGPU(matrixB, buffer);
    testScalarGPU(testOut, buffer, 0.0); // -1.0 + 0.5 + 2.0 - 1.5
  });
  test('embeddingLookupGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> indices = GPUTensor<Vector>([1.0, 0.0]); // Lookup row 1, then row 0
    GPUTensor<Matrix> testOut = embeddingLookupGPU(indices, matrixA, buffer);
    testMatrixGPU(testOut, buffer, [
      [3.0, 0.0],
      [1.0, -2.0],
    ]);
  });
  test('embeddingLookupBatchGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> batchIndices = GPUTensor<Matrix>([
      [1.0, 0.0],
      [0.0, 1.0],
    ]);
    GPUTensor<Tensor3D> testOut = embeddingLookupBatchGPU(batchIndices, matrixA, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [3.0, 0.0],
        [1.0, -2.0],
      ],
      [
        [1.0, -2.0],
        [3.0, 0.0],
      ],
    ]);
  });
  test('sumReduceColumnsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = sumReduceColumnsGPU(matrixB, buffer);
    testVectorGPU(testOut, buffer, [1.0, -1.0]); // Col 0 sum, Col 1 sum
  });
  test('sumReduceRowsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = sumReduceRowsGPU(matrixB, buffer);
    testVectorGPU(testOut, buffer, [-0.5, 0.5]); // Row 0 sum, Row 1 sum
  });

  test('sliceColumnGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = sliceColumnGPU(matrixA, 0, 1, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0],
      [3.0],
    ]);
  });
  test('selectRowGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = selectRowGPU(matrixB, 1, buffer);
    testVectorGPU(testOut, buffer, [2.0, -1.5]);
  });

  test('selectMatrixFrom3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = selectMatrixFrom3DGPU(tensor3DA, 0, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });
  test('concatenateGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> testOut = concatenateGPU(vectorA, vectorB, buffer);
    testVectorGPU(testOut, buffer, [-1.0, 0.0, 2.0, 0.5, -1.5, 3.0]);
  });
  test('concatenateMatricesByColumnGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = concatenateMatricesByColumnGPU([matrixA, matrixB], buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, -2.0, -1.0, 0.5],
      [3.0, 0.0, 2.0, -1.5],
    ]);
  });
  test('concatenate3DGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = concatenate3DGPU(tensor3DA, tensor3DB, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
      [
        [2.0, 3.0],
        [4.0, 5.0],
      ],
    ]);
  });

  test('stackMatricesGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> testOut = stackMatricesGPU([matrixA, matrixB], buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, -2.0],
        [3.0, 0.0],
      ],
      [
        [-1.0, 0.5],
        [2.0, -1.5],
      ],
    ]);
  });
  test('scatterHeadsGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = scatterHeadsGPU([matrixA, matrixB], 4, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, -2.0, -1.0, 0.5],
      [3.0, 0.0, 2.0, -1.5],
    ]);
  });
  test('padMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> testOut = padMatrixGPU(matrixC, 1, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, -2.0, 3.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
    ]);
  });

  test('conv2dMultiChannelGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    GPUTensor<Tensor3D> weight = GPUTensor<Tensor3D>([
      [
        [1.0, 0.0],
        [0.0, -1.0],
      ],
    ]);
    GPUTensor<Vector> bias = GPUTensor<Vector>([0.0]);

    GPUTensor<Tensor3D> testOut = conv2dMultiChannelGPU(input, weight, bias, 2, 2, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [-4.0, -4.0],
        [-4.0, -4.0],
      ],
    ]);
  });
  test('conv2dSimpleGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    GPUTensor<Matrix> kernel = GPUTensor<Matrix>([
      [1.0, 0.0],
      [0.0, -1.0],
    ]);

    GPUTensor<Matrix> testOut = conv2dSimpleGPU(input, kernel, buffer);
    testMatrixGPU(testOut, buffer, [
      [-4.0, -4.0],
      [-4.0, -4.0],
    ]);
  });

  test('im2colGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);

    GPUTensor<Matrix> testOut = im2colGPU(input, 2, 2, buffer);
    testMatrixGPU(testOut, buffer, [
      [1.0, 2.0, 4.0, 5.0],
      [2.0, 3.0, 5.0, 6.0],
      [4.0, 5.0, 7.0, 8.0],
      [5.0, 6.0, 8.0, 9.0],
    ]);
  });

  test('maxPool1dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> input = GPUTensor<Vector>([1.0, 3.0, 2.0, 5.0, 4.0]);

    GPUTensor<Vector> testOut = maxPool1dGPU(input, 2, 2, buffer);
    testVectorGPU(testOut, buffer, [3.0, 5.0]);
  });
  test('maxPool2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ]);

    GPUTensor<Matrix> testOut = maxPool2dGPU(input, 2, 2, buffer);
    testMatrixGPU(testOut, buffer, [
      [6.0, 8.0],
      [14.0, 16.0],
    ]);
  });
  test('avgPool2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ]);

    GPUTensor<Matrix> testOut = avgPool2dGPU(input, 2, 2, buffer);
    testMatrixGPU(testOut, buffer, [
      [3.5, 5.5],
      [11.5, 13.5],
    ]);
  });
  test('globalAveragePoolingGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);

    GPUTensor<Vector> testOut = globalAveragePoolingGPU(input, buffer);
    testVectorGPU(testOut, buffer, [3.0, 4.0]);
  });

  test('batchNorm1dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> x = GPUTensor<Vector>([2.0, 4.0]);
    GPUTensor<Vector> gamma = GPUTensor<Vector>([1.0, 2.0]);
    GPUTensor<Vector> beta = GPUTensor<Vector>([0.0, 1.0]);
    GPUTensor<Vector> runningMean = GPUTensor<Vector>([0.0, 2.0]);
    GPUTensor<Vector> runningVar = GPUTensor<Vector>([4.0, 4.0]);

    GPUTensor<Vector> testOut = batchNorm1dGPU(
      x, gamma, beta, runningMean, runningVar, 0.9, 0.0, false, buffer,
    );
    testVectorGPU(testOut, buffer, [1.0, 3.0]);
  });
  test('batchNorm2dGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> x = GPUTensor<Tensor3D>([
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
    ]);
    GPUTensor<Vector> gamma = GPUTensor<Vector>([2.0]);
    GPUTensor<Vector> beta = GPUTensor<Vector>([1.0]);
    GPUTensor<Vector> runningMean = GPUTensor<Vector>([2.5]);
    GPUTensor<Vector> runningVar = GPUTensor<Vector>([1.0]);

    GPUTensor<Tensor3D> testOut = batchNorm2dGPU(
      x, gamma, beta, runningMean, runningVar, 0.9, 0.0, false, buffer,
    );
    testTensor3DGPU(testOut, buffer, [
      [
        [-2.0, 0.0],
        [2.0, 4.0],
      ],
    ]);
  });
  test('layerNormMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> m = GPUTensor<Matrix>([
      [1.0, 3.0],
      [2.0, 4.0],
    ]);
    GPUTensor<Vector> gamma = GPUTensor<Vector>([1.0, 1.0]);
    GPUTensor<Vector> beta = GPUTensor<Vector>([0.0, 0.0]);
    GPUTensor<Vector> meanCache = GPUTensor<Vector>([0.0, 0.0]);
    GPUTensor<Vector> rstdCache = GPUTensor<Vector>([0.0, 0.0]);

    GPUTensor<Matrix> testOut = layerNormMatrixGPU(
      m, gamma, beta, meanCache, rstdCache, 0.0, buffer,
    );
    testMatrixGPU(testOut, buffer, [
      [-1.0, 1.0],
      [-1.0, 1.0],
    ]);
  });

  test('dropoutGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> input = GPUTensor<Vector>([1.0, 2.0, 3.0]);

    // Testing with 0.0 rate to ensure deterministic pass-through
    GPUTensor<Vector> testOut = dropoutGPU(input, 0.0, buffer);
    testVectorGPU(testOut, buffer, [1.0, 2.0, 3.0]);
  });

  test('buildMarkovTableGPU', () {
    CommandBuffer buffer = CommandBuffer();
    // Sequence: 0 -> 1 -> 0 -> 1 -> 1
    // Transitions from 0: to 1 (twice). Probs: [0.0, 1.0]
    // Transitions from 1: to 0 (once), to 1 (once). Probs: [0.5, 0.5]
    GPUTensor<Vector> sequence = GPUTensor<Vector>([0.0, 1.0, 0.0, 1.0, 1.0]);

    GPUTensor<Matrix> testOut = buildMarkovTableGPU(sequence, 1, 2, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.0, 1.0],
      [0.5, 0.5],
    ]);
  });
  test('markovPredictGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> historyBatch = GPUTensor<Matrix>([
      [1.0], // Request prediction for history 1
      [0.0], // Request prediction for history 0
    ]);
    GPUTensor<Matrix> probTable = GPUTensor<Matrix>([
      [0.0, 1.0], // Probs for history 0
      [0.5, 0.5], // Probs for history 1
    ]);

    GPUTensor<Matrix> testOut = markovPredictGPU(historyBatch, probTable, 2, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.5, 0.5], // Looked up row 1
      [0.0, 1.0], // Looked up row 0
    ]);
  });

  test('matMulBiasReluGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> x = GPUTensor<Matrix>([
      [1.0, -2.0],
      [3.0, 0.0],
    ]);
    GPUTensor<Matrix> w = GPUTensor<Matrix>([
      [-1.0, 2.0],
      [0.5, -1.5],
    ]);
    GPUTensor<Vector> b = GPUTensor<Vector>([1.0, -1.0]);
    List<GPUTensor> intermediates = <GPUTensor>[];

    // 1. X @ W = [[-2.0, 5.0], [-3.0, 6.0]]
    // 2. Add Bias = [[-1.0, 4.0], [-2.0, 5.0]]
    // 3. ReLU = [[0.0, 4.0], [0.0, 5.0]]
    GPUTensor<Matrix> testOut = matMulBiasReluGPU(x, w, b, buffer, intermediates);
    testMatrixGPU(testOut, buffer, [
      [0.0, 4.0],
      [0.0, 5.0],
    ]);

    // Verify the intermediate tensor was correctly stored for VRAM lifecycle management
    expect(intermediates.length, 1);
  });

  test('dotProductGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>([1.0, 2.0, 3.0]);
    GPUTensor<Vector> b = GPUTensor<Vector>([2.0, -1.0, 0.5]);

    // 1*2 + 2*(-1) + 3*0.5 = 2 - 2 + 1.5 = 1.5
    GPUTensor<Scalar> testOut = dotProductGPU(a, b, buffer);
    testScalarGPU(testOut, buffer, 1.5);
  });


  test('l2NormGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> v = GPUTensor<Vector>([3.0, -4.0]);

    // sqrt(3^2 + (-4)^2) = sqrt(9 + 16) = sqrt(25) = 5.0
    GPUTensor<Scalar> testOut = l2NormGPU(v, buffer);
    testScalarGPU(testOut, buffer, 5.0);
  });
  test('euclideanDistanceGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>([1.0, 2.0, 3.0]);
    GPUTensor<Vector> b = GPUTensor<Vector>([4.0, 2.0, -1.0]);

    // diff = [-3.0, 0.0, 4.0]
    // squared = [9.0, 0.0, 16.0] -> sum = 25.0 -> sqrt = 5.0
    GPUTensor<Scalar> testOut = euclideanDistanceGPU(a, b, buffer);
    testScalarGPU(testOut, buffer, 5.0);
  });
  test('cosineSimilarityGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> a = GPUTensor<Vector>([3.0, 4.0]); // norm = 5
    GPUTensor<Vector> b = GPUTensor<Vector>([6.0, 8.0]); // norm = 10

    // dot = 18 + 32 = 50. denom = 5 * 10 = 50. sim = 50/50 = 1.0
    GPUTensor<Scalar> testOut = cosineSimilarityGPU(a, b, buffer);
    testScalarGPU(testOut, buffer, 1.0);
  });

  test('maeLossGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Vector> preds = GPUTensor<Vector>([1.0, 3.0, 5.0]);
    GPUTensor<Vector> targets = GPUTensor<Vector>([2.0, 1.0, 5.0]);

    // diff = [-1.0, 2.0, 0.0] -> abs = [1.0, 2.0, 0.0] -> sum = 3.0
    // N = 3.0 -> mean = 1.0
    GPUTensor<Scalar> testOut = maeLossGPU(preds, targets, buffer);
    testScalarGPU(testOut, buffer, 1.0);
  });
  test('rmsNormMatrixGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> input = GPUTensor<Matrix>([
      [3.0, 4.0],
    ]);
    GPUTensor<Vector> weight = GPUTensor<Vector>([1.0, 1.0]);

    // RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) = 3.5355339
    // Normalized = [3/3.5355, 4/3.5355] = [0.848528, 1.13137]
    GPUTensor<Matrix> testOut = rmsNormMatrixGPU(input, weight, 0.0, false, buffer);
    testMatrixGPU(testOut, buffer, [
      [0.848528, 1.131371],
    ]);
  });
  test('causalMaskGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>([
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
    ]);

    // Batch=1, Heads=1, SeqLen=2
    // Elements where j > i are masked to -10000.0 (Row 0, Col 1 is masked)
    GPUTensor<Tensor3D> testOut = causalMaskGPU(input, 1, 1, 2, buffer);
    testTensor3DGPU(testOut, buffer, [
      [
        [1.0, -10000.0],
        [3.0, 4.0],
      ],
    ]);
  });
  test('applyRopeGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Tensor3D> input = GPUTensor<Tensor3D>([
      [
        [1.0, 2.0],
      ],
    ]);

    // Representing cos(90deg) and sin(90deg)
    GPUTensor<Matrix> cosTable = GPUTensor<Matrix>([[0.0]]);
    GPUTensor<Matrix> sinTable = GPUTensor<Matrix>([[1.0]]);

    // Batch=1, Seq=1, Heads=1, HeadDim=2
    // out[0] = 1.0*0.0 - 2.0*1.0 = -2.0
    // out[1] = 2.0*0.0 + 1.0*1.0 = 1.0
    GPUTensor<Tensor3D> testOut = applyRopeGPU(
        input, cosTable, sinTable, 1, 1, 1, 2, buffer);

    testTensor3DGPU(testOut, buffer, [
      [
        [-2.0, 1.0],
      ],
    ]);
  });
  test('crossEntropyLossGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> logits = GPUTensor<Matrix>([
      [0.0, 1.0, 0.0],
    ]);
    GPUTensor<Vector> targets = GPUTensor<Vector>([1.0]); // Target index is 1

    // Max = 1.0
    // Exp = [exp(-1), exp(0), exp(-1)] = [0.367879, 1.0, 0.367879] -> Sum = 1.735758
    // Target Logit = 1.0
    // Loss = log(1.735758) - (1.0 - 1.0) = 0.551444
    GPUTensor<Vector> testOut = crossEntropyLossGPU(logits, targets, buffer);
    testVectorGPU(testOut, buffer, [0.551444]);
  });
  test('argmaxGPU', () {
    CommandBuffer buffer = CommandBuffer();
    GPUTensor<Matrix> logits = GPUTensor<Matrix>([
      [1.0, 5.0, 2.0],
      [8.0, -1.0, 3.0],
    ]);

    // Row 0 max is 5.0 at index 1. Row 1 max is 8.0 at index 0.
    GPUTensor<Vector> testOut = argmaxGPU(logits, buffer);
    testVectorGPU(testOut, buffer, [1.0, 0.0]);
  });
}
