import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  // setUpAll runs exactly once before any tests start.
  setUpAll(() async {});

  Tensor<Scalar> scalarA = Tensor<Scalar>(1.0);
  Tensor<Scalar> scalarB = Tensor<Scalar>(2.0);
  Tensor<Scalar> scalarC = Tensor<Scalar>(3.0);

  Tensor<Vector> vectorA = Tensor<Vector>([-1.0, -2.0, -3.0]);
  Tensor<Vector> vectorB = Tensor<Vector>([1.0, 0.0, -1.0]);
  Tensor<Vector> vectorC = Tensor<Vector>([1.0, 2.0, 3.0]);
  Tensor<Vector> vectorD = Tensor<Vector>([1.0, 2.0]);

  Tensor<Matrix> matrixA = Tensor<Matrix>([
    [-1.0, -2.0],
    [-1.0, -2.0],
  ]);
  Tensor<Matrix> matrixB = Tensor<Matrix>([
    [1.0, 2.0],
    [1.0, 2.0],
  ]);
  Tensor<Matrix> matrixC = Tensor<Matrix>([
    [2.0, 4.0],
    [2.0, 4.0],
  ]);

  Tensor<Tensor3D> tensor3DA = Tensor<Tensor3D>([
    [
      [1.0, 2.0],
      [3.0, 4.0],
    ],
  ]);
  Tensor<Tensor3D> tensor3DB = Tensor<Tensor3D>([
    [
      [2.0, 3.0],
      [4.0, 5.0],
    ],
  ]);

  testVector(List<double> actual, List<double> wanted) {
    for (int i = 0; i < actual.length; i++) {
      expect(actual[i], closeTo(wanted[i], 1e-5));
    }
  }

  testMatrix(List<List<double>> actual, List<List<double>> wanted) {
    for (int i = 0; i < actual.length; i++) {
      for (int j = 0; j < actual[i].length; j++) {
        expect(actual[i][j], closeTo(wanted[i][j], 1e-5));
      }
    }
  }

  testTensor3D(
    List<List<List<double>>> actual,
    List<List<List<double>>> wanted,
  ) {
    for (int i = 0; i < actual.length; i++) {
      for (int j = 0; j < actual[i].length; j++) {
        for (int k = 0; k < actual[i][j].length; k++) {
          expect(actual[i][j][k], closeTo(wanted[i][j][k], 1e-5));
        }
      }
    }
  }

  test('Add', () {
    Tensor<Scalar> test = add(scalarA, scalarB);
    expect(test.value, closeTo(3.0, 1e-5));
  });

  test('Multiply', () {
    Tensor<Scalar> test = multiply(scalarA, scalarB);
    expect(test.value, closeTo(2.0, 1e-5));
  });

  test('sigmoidScalar', () {
    Tensor<Scalar> test = sigmoidScalar(scalarC);
    expect(test.value, closeTo(0.9525741, 1e-5));
  });
  test('binaryCrossEntropy', () {
    Tensor<Scalar> small = Tensor<Scalar>(0.7);
    Tensor<Scalar> big = Tensor<Scalar>(0.9);
    Tensor<Scalar> test = binaryCrossEntropy(small, big);
    expect(test.value, closeTo(0.44140473, 1e-5));
  });
  test('addVector', () {
    Tensor<Vector> test = addVector(vectorA, vectorB);
    testVector(test.value, [0.0, -2.0, -4.0]);
  });
  test('addScalar', () {
    Tensor<Vector> test = addScalar(vectorA, 2.0);
    testVector(test.value, [1.0, -0.0, -1.0]);
  });
  test('addScalar', () {
    Tensor<Vector> test = addScalar(vectorA, 2.0);
    testVector(test.value, [1.0, -0.0, -1.0]);
  });
  test('concatenate', () {
    Tensor<Vector> test = concatenate(vectorA, vectorB);
    testVector(test.value, [-1.0, -2.0, -3.0, 1.0, 0.0, -1.0]);
  });
  test('dorProduct', () {
    Tensor<Scalar> test = dot(vectorA, vectorB);
    expect(test.value, closeTo(02.0, 1e-5));
  });
  test('elementWiseMultiply', () {
    Tensor<Vector> test = elementWiseMultiply(vectorA, vectorB);
    testVector(test.value, [-1.0, 0.0, 3.0]);
  });
  test('meanSquareError', () {
    Tensor<Scalar> test = mse(vectorA, vectorB);
    expect(test.value, closeTo(04.0, 1e-5));
  });
  test('relu', () {
    Tensor<Vector> test = relu(vectorB);
    testVector(test.value, [1.0, 0.0, 0.0]);
  });
  test('sigmoid', () {
    Tensor<Vector> test = sigmoid(vectorC);
    testVector(test.value, [0.731058, 0.880797, 0.95257]);
  });
  test('sum', () {
    Tensor<Scalar> test = sum(vectorC);
    expect(test.value, closeTo(06.0, 1e-5));
  });
  test('tanh', () {
    Tensor<Vector> test = vectorTanh(vectorB);
    testVector(test.value, [0.76159, 0.0, -0.76159]);
  });
  test('vectorExp', () {
    Tensor<Vector> test = vectorExp(vectorC);
    testVector(test.value, [2.71828, 7.38905, 20.08553]);
  });
  test('vectorLog', () {
    Tensor<Vector> test = vectorLog(vectorC);
    testVector(test.value, [0, 0.69314, 1.09861]);
  });
  test('softplus', () {
    Tensor<Vector> test = softplus(vectorC);
    testVector(test.value, [1.3132617, 2.126928, 3.048587]);
  });
  test('avgPool1d', () {
    Tensor<Vector> test = avgPool1d(vectorC, 2, 1);
    testVector(test.value, [1.5, 2.5]);
  });

  test('addMatrix', () {
    Tensor<Matrix> test = addMatrix(matrixA, matrixC);
    testMatrix(test.value, [
      [1.0, 2.0],
      [1.0, 2.0],
    ]);
  });
  test('addMatrixAndVector', () {
    Tensor<Matrix> test = addMatrixAndVector(matrixA, vectorD);
    testMatrix(test.value, [
      [0.0, 0.0],
      [0.0, 0.0],
    ]);
  });
  test('addScalarToMatrix', () {
    Tensor<Matrix> test = addScalarToMatrix(matrixA, scalarA);
    testMatrix(test.value, [
      [0.0, -1.0],
      [0.0, -1.0],
    ]);
  });
  test('addScalarToMatrix', () {
    Tensor<Matrix> test = addScalarToMatrix(matrixA, scalarA);
    testMatrix(test.value, [
      [0.0, -1.0],
      [0.0, -1.0],
    ]);
  });
  test('concatenateMatricesByColumn', () {
    List<Tensor<Matrix>> list = [matrixA, matrixB];
    Tensor<Matrix> testOut = concatenateMatricesByColumn(list);
    testMatrix(testOut.value, [
      [-1.0, -2.0, 1.0, 2.0],
      [-1.0, -2.0, 1.0, 2.0],
    ]);
  });
  test('elementWiseMultiplyMatrix', () {
    Tensor<Matrix> testOut = elementWiseMultiplyMatrix(matrixA, matrixB);
    testMatrix(testOut.value, [
      [-1.0, -4.0],
      [-1.0, -4.0],
    ]);
  });
  test('conv2d_valid', () {
    Tensor<Matrix> input = Tensor<Matrix>([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    Tensor<Matrix> kernel = Tensor<Matrix>([
      [1.0, 0.0],
      [0.0, -1.0],
    ]);
    Tensor<Matrix> testOut = conv2d(input, kernel, padding: 'valid');
    testMatrix(testOut.value, [
      [-4.0, -4.0],
      [-4.0, -4.0],
    ]);
  });
  test('conv2d_same', () {
    Tensor<Matrix> kernel = Tensor<Matrix>([
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0],
    ]);
    Tensor<Matrix> testOut = conv2d(matrixB, kernel, padding: 'same');
    testMatrix(testOut.value, [
      [1.0, 2.0],
      [1.0, 2.0],
    ]);
  });
  test('matMul', () {
    Tensor<Matrix> testOut = matMul(matrixA, matrixB);
    testMatrix(testOut.value, [
      [-3.0, -6.0],
      [-3.0, -6.0],
    ]);
  });
  test('matVecMul', () {
    Tensor<Vector> testOut = matVecMul(matrixA, vectorD);
    testVector(testOut.value, [-5.0, -5.0]);
  });
  test('mseMatrix', () {
    Tensor<Scalar> testOut = mseMatrix(matrixA, matrixB);
    expect(testOut.value, closeTo(10.0, 1e-5));
  });
  test('reluMatrix', () {
    Tensor<Matrix> testOut = reluMatrix(matrixA);
    testMatrix(testOut.value, [
      [0.0, 0.0],
      [0.0, 0.0],
    ]);
  });
  test('reshapeVectorToMatrix', () {
    Tensor<Vector> vector = Tensor<Vector>([1.0, 2.0, 3.0, 4.0]);
    Tensor<Matrix> testOut = reshapeVectorToMatrix(vector, 2, 2);
    testMatrix(testOut.value, [
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });
  test('scaleMatrix', () {
    Tensor<Matrix> testOut = scaleMatrix(matrixB, 2);
    testMatrix(testOut.value, matrixC.value);
  });
  test('selectRow', () {
    Tensor<Vector> testOut = selectRow(matrixB, 1);
    testVector(testOut.value, [1.0, 2.0]);
  });
  test('sigmoidMatrix', () {
    Tensor<Matrix> testOut = sigmoidMatrix(matrixB);
    testMatrix(testOut.value, [
      [0.731058, 0.880797],
      [0.731058, 0.880797],
    ]);
  });
  test('sumMatrix', () {
    Tensor<Scalar> testOut = sumMatrix(matrixB);
    expect(testOut.value, closeTo(6.0, 1e-5));
  });
  test('tanhMatrix', () {
    Tensor<Matrix> testOut = tanhMatrix(matrixB);
    testMatrix(testOut.value, [
      [0.761594, 0.964027],
      [0.761594, 0.964027],
    ]);
  });
  test('transpose', () {
    Tensor<Matrix> testOut = transpose(matrixB);
    testMatrix(testOut.value, [
      [1.0, 1.0],
      [2.0, 2.0],
    ]);
  });
  test('softmaxMatrix', () {
    Tensor<Matrix> testOut = softmaxMatrix(matrixB);
    testMatrix(testOut.value, [
      [0.26894, 0.73106],
      [0.26894, 0.73106],
    ]);
  });
  test('avgPool2d', () {
    Tensor<Matrix> input = Tensor<Matrix>([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ]);
    Tensor<Matrix> testOut = avgPool2d(input, 2, 2);
    testMatrix(testOut.value, [
      [3.5, 5.5],
      [11.5, 13.5],
    ]);
  });
  test('globalAveragePooling', () {
    Tensor<Matrix> input = Tensor<Matrix>([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);
    Tensor<Vector> testOut = globalAveragePooling(input);
    testVector(testOut.value, [3.0, 4.0]);
  });
  test('batchNorm1dMath_inference', () {
    Tensor<Vector> x = Tensor<Vector>([2.0, 4.0]);
    Tensor<Vector> gamma = Tensor<Vector>([1.0, 2.0]);
    Tensor<Vector> beta = Tensor<Vector>([0.0, 1.0]);
    Vector runningMean = [0.0, 2.0];
    Vector runningVar = [4.0, 4.0];

    Tensor<Vector> testOut = batchNorm1dMath(
      x,
      gamma,
      beta,
      runningMean,
      runningVar,
      2,
      false,
      0.9,
      0.0,
    );
    testVector(testOut.value, [1.0, 3.0]);
  });
  test('batchNorm2dMath_inference', () {
    Tensor<Tensor3D> x = Tensor<Tensor3D>([
      [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
    ]);
    Tensor<Vector> gamma = Tensor<Vector>([2.0]);
    Tensor<Vector> beta = Tensor<Vector>([1.0]);
    Vector runningMean = [2.5];
    Vector runningVar = [1.0];

    Tensor<Tensor3D> testOut = batchNorm2dMath(
      x,
      gamma,
      beta,
      runningMean,
      runningVar,
      1,
      false,
      0.9,
      0.0,
    );
    testTensor3D(testOut.value, [
      [
        [-2.0, 0.0],
        [2.0, 4.0],
      ],
    ]);
  });

  test('add3D', () {
    Tensor<Tensor3D> testOut = add3D(tensor3DA, tensor3DB);
    testTensor3D(testOut.value, [
      [
        [3.0, 5.0],
        [7.0, 9.0],
      ],
    ]);
  });
  test('elementWiseMultiply3D', () {
    Tensor<Tensor3D> testOut = elementWiseMultiply3D(tensor3DA, tensor3DB);
    testTensor3D(testOut.value, [
      [
        [2.0, 6.0],
        [12.0, 20.0],
      ],
    ]);
  });
  test('concatenate3D', () {
    Tensor<Tensor3D> testOut = concatenate3D(tensor3DA, tensor3DB);
    testTensor3D(testOut.value, [
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
  test('stackMatricesTo3D', () {
    List<Tensor<Matrix>> list = [matrixA, matrixB];
    Tensor<Tensor3D> testOut = stackMatricesTo3D(list);
    testTensor3D(testOut.value, [
      [
        [-1.0, -2.0],
        [-1.0, -2.0],
      ],
      [
        [1.0, 2.0],
        [1.0, 2.0],
      ],
    ]);
  });

  /// Impossible to test in training mode due to randomness
  test('dropoutVectorMath_inference', () {
    Tensor<Vector> testOut = dropoutVectorMath(vectorC, 0.5, false);
    testVector(testOut.value, [1.0, 2.0, 3.0]);
  });
  test('dropoutMatrixMath_inference', () {
    Tensor<Matrix> testOut = dropoutMatrixMath(matrixB, 0.5, false);
    testMatrix(testOut.value, [
      [1.0, 2.0],
      [1.0, 2.0],
    ]);
  });

  test('maxPool1d', () {
    Tensor<Vector> input = Tensor<Vector>([1.0, 3.0, 2.0, 5.0, 4.0]);
    Tensor<Vector> testOut = maxPool1d(input, 2, 2);
    testVector(testOut.value, [3.0, 5.0]);
  });
  test('maxPool2d', () {
    Tensor<Matrix> input = Tensor<Matrix>([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ]);
    Tensor<Matrix> testOut = maxPool2d(input, 2, 2);
    testMatrix(testOut.value, [
      [6.0, 8.0],
      [14.0, 16.0],
    ]);
  });

  test('softmaxVector', () {
    Tensor<Vector> testOut = softmaxVector(vectorC);
    testVector(testOut.value, [0.09003, 0.24472, 0.66524]);
  });
  test('swishVector', () {
    Tensor<Vector> testOut = swishVector(vectorB);
    testVector(testOut.value, [0.731058, 0.0, -0.268941]);
  });
  test('swishMatrix', () {
    Tensor<Matrix> testOut = swishMatrix(matrixB);
    testMatrix(testOut.value, [
      [0.731058, 1.761594],
      [0.731058, 1.761594],
    ]);
  });
  test('eluVector', () {
    Tensor<Vector> testOut = eluVector(vectorB, 1.0);
    testVector(testOut.value, [1.0, 0.0, -0.63212]);
  });
  test('eluMatrix', () {
    Tensor<Matrix> testOut = eluMatrix(matrixA, 1.0);
    testMatrix(testOut.value, [
      [-0.63212, -0.86466],
      [-0.63212, -0.86466],
    ]);
  });
  test('leakyReluVector', () {
    Tensor<Vector> testOut = leakyReluVector(vectorB, 0.1);
    testVector(testOut.value, [1.0, 0.0, -0.1]);
  });
  test('leakyReluMatrix', () {
    Tensor<Matrix> testOut = leakyReluMatrix(matrixA, 0.1);
    testMatrix(testOut.value, [
      [-0.1, -0.2],
      [-0.1, -0.2],
    ]);
  });
  test('mishVector', () {
    Tensor<Vector> testOut = mishVector(vectorB);
    testVector(testOut.value, [0.865098, 0.0, -0.30340]);
  });
  test('mishMatrix', () {
    Tensor<Matrix> testOut = mishMatrix(matrixB);
    testMatrix(testOut.value, [
      [0.865098, 1.94395],
      [0.865098, 1.94395],
    ]);
  });
}
