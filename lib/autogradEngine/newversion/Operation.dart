import 'executer.dart';

/// The abstract base class for all mathematical operations in the graph.
abstract class Operation {
  String get opName;
  dynamic forward(List<dynamic> inputValues);
  List<dynamic> backward(dynamic outputGrad, List<dynamic> inputValues);
}

/// A concrete implementation of the 'matMul' operation.
class MatMulOp extends Operation {
  @override
  String get opName => 'matMul';

  @override
  dynamic forward(List<dynamic> inputValues) {
    Matrix a = inputValues[0] as Matrix;
    Matrix b = inputValues[1] as Matrix;
    int M = a.length;
    int N = a[0].length;
    int P = b[0].length;
    Matrix outValue = [];
    for (int i = 0; i < M; i++) {
      Vector outRow = List.filled(P, 0.0);
      for (int j = 0; j < P; j++) {
        for (int k = 0; k < N; k++) {
          outRow[j] += a[i][k] * b[k][j];
        }
      }
      outValue.add(outRow);
    }
    return outValue;
  }

  @override
  List<dynamic> backward(dynamic outputGrad, List<dynamic> inputValues) {
    Matrix a_val = inputValues[0] as Matrix;
    Matrix b_val = inputValues[1] as Matrix;
    Matrix grad = outputGrad as Matrix;

    Matrix bT = _transpose(b_val);
    Matrix a_grad = _matMul(grad, bT);

    Matrix aT = _transpose(a_val);
    Matrix b_grad = _matMul(aT, grad);

    return [a_grad, b_grad];
  }

  // Private helpers for math, could be moved to a utility file
  Matrix _transpose(Matrix m) {
    int rows = m.length;
    int cols = m[0].length;
    Matrix transposed = [];
    for(int c=0; c<cols; c++){
      Vector newRow = [];
      for(int r=0; r<rows; r++){
        newRow.add(m[r][c]);
      }
      transposed.add(newRow);
    }
    return transposed;
  }

  Matrix _matMul(Matrix a, Matrix b) {
    int M = a.length;
    int N = a[0].length;
    int P = b[0].length;
    Matrix outValue = [];
    for (int i = 0; i < M; i++) {
      Vector outRow = List.filled(P, 0.0);
      for (int j = 0; j < P; j++) {
        for (int k = 0; k < N; k++) {
          outRow[j] += a[i][k] * b[k][j];
        }
      }
      outValue.add(outRow);
    }
    return outValue;
  }
}

/// A concrete implementation of the 'addMatrixAndVector' operation.
class AddMatrixAndVectorOp extends Operation {
  @override
  String get opName => 'addMatrixAndVector';

  @override
  dynamic forward(List<dynamic> inputValues) {
    Matrix m = inputValues[0] as Matrix;
    Vector v = inputValues[1] as Vector;
    Matrix outValue = [];
    for (int i = 0; i < m.length; i++) {
      Vector row = [];
      for (int j = 0; j < m[0].length; j++) {
        row.add(m[i][j] + v[j]);
      }
      outValue.add(row);
    }
    return outValue;
  }

  @override
  List<dynamic> backward(dynamic outputGrad, List<dynamic> inputValues) {
    Matrix gradMatrix = outputGrad as Matrix;
    Vector v = inputValues[1] as Vector;

    Matrix m_grad = gradMatrix; // Gradient for matrix flows straight through

    Vector v_grad = List.filled(v.length, 0.0);
    for (int j = 0; j < gradMatrix[0].length; j++) {
      for (int i = 0; i < gradMatrix.length; i++) {
        v_grad[j] += gradMatrix[i][j];
      }
    }
    return [m_grad, v_grad];
  }
}