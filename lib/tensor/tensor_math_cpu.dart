import 'dart:math';
import 'type_Aliases.dart';

import '../tensor/tensor.dart';

// ─────────────────────────────────────────────────────── //
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────── //

Matrix padMatrix(Matrix input, int padding) {
  int inputHeight = input.length;
  int inputWidth = input[0].length;
  int newHeight = inputHeight + 2 * padding;
  int newWidth  = inputWidth + 2 * padding;

  Matrix padded = [];
  for (int i = 0; i < newHeight; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < newWidth; j = j + 1) {
      row.add(0.0);
    }
    padded.add(row);
  }

  for (int i = 0; i < inputHeight; i = i + 1) {
    for (int j = 0; j < inputWidth; j = j + 1) {
      padded[i + padding][j + padding] = input[i][j];
    }
  }
  return padded;
}

// ─────────────────────────────────────────────────────── //
// SCALAR (0D) OPERATIONS
// ─────────────────────────────────────────────────────── //

Tensor<Scalar> add(Tensor<Scalar> a, Tensor<Scalar> b) {
  Tensor<Scalar> out = Tensor<Scalar>(a.data[0] + b.data[0]);

  out.creator = Node(
    [a, b],
        () {
      a.grad[0] = a.grad[0] + out.grad[0];
      b.grad[0] = b.grad[0] + out.grad[0];
    },
    opName: 'add',
    cost: 1,
  );
  return out;
}

Tensor<Scalar> multiply(Tensor<Scalar> a, Tensor<Scalar> b) {
  Tensor<Scalar> out = Tensor<Scalar>(a.data[0] * b.data[0]);

  out.creator = Node(
    [a, b],
        () {
      a.grad[0] = a.grad[0] + out.grad[0] * b.data[0];
      b.grad[0] = b.grad[0] + out.grad[0] * a.data[0];
    },
    opName: 'multiply_scalar',
    cost: 1,
  );
  return out;
}

Tensor<Scalar> sigmoidScalar(Tensor<Scalar> s) {
  double val = 1.0 / (1.0 + exp(-s.data[0]));
  Tensor<Scalar> out = Tensor<Scalar>(val);

  out.creator = Node(
    [s],
        () {
      s.grad[0] = s.grad[0] + out.grad[0] * (val * (1.0 - val));
    },
    opName: 'sigmoidScalar',
    cost: 1,
  );
  return out;
}

Tensor<Scalar> binaryCrossEntropy(Tensor<Scalar> prediction, Tensor<Scalar> target) {
  double predVal = prediction.data[0];
  double targetVal = target.data[0];

  double outValue = -(targetVal * log(predVal) + (1.0 - targetVal) * log(1.0 - predVal));
  Tensor<Scalar> out = Tensor<Scalar>(outValue);

  out.creator = Node(
    [prediction, target],
        () {
      prediction.grad[0] = prediction.grad[0] +
          out.grad[0] * ((predVal - targetVal) / (predVal * (1.0 - predVal)));
    },
    opName: 'binaryCrossEntropy',
    cost: 1,
  );
  return out;
}

// ─────────────────────────────────────────────────────── //
// VECTOR (1D) OPERATIONS
// ─────────────────────────────────────────────────────── //

Tensor<Vector> addVector(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(a.data[i] + b.data[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < N; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i];
        b.grad[i] = b.grad[i] + out.grad[i];
      }
    },
    opName: 'add_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> addScalar(Tensor<Vector> v, double s) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(v.data[i] + s);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[i];
      }
    },
    opName: 'addScalar_vector',
    extraParams: {'s': s},
    cost: N,
  );
  return out;
}

Tensor<Vector> concatenate(Tensor<Vector> a, Tensor<Vector> b) {
  int aLength = a.data.length;
  int bLength = b.data.length;
  Vector outValue = [];

  for (int i = 0; i < aLength; i = i + 1) {
    outValue.add(a.data[i]);
  }
  for (int i = 0; i < bLength; i = i + 1) {
    outValue.add(b.data[i]);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < aLength; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i];
      }
      for (int i = 0; i < bLength; i = i + 1) {
        b.grad[i] = b.grad[i] + out.grad[aLength + i];
      }
    },
    opName: 'concat_vector',
    cost: 0,
  );
  return out;
}

Tensor<Scalar> dot(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.data.length;
  double outValue = 0.0;
  for (int i = 0; i < N; i = i + 1) {
    outValue = outValue + (a.data[i] * b.data[i]);
  }
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < N; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[0] * b.data[i];
        b.grad[i] = b.grad[i] + out.grad[0] * a.data[i];
      }
    },
    opName: 'dot',
    cost: 2 * N,
  );
  return out;
}

Tensor<Vector> elementWiseMultiply(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(a.data[i] * b.data[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < N; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i] * b.data[i];
        b.grad[i] = b.grad[i] + out.grad[i] * a.data[i];
      }
    },
    opName: 'multiply_vector',
    cost: N,
  );
  return out;
}

Tensor<Scalar> mse(Tensor<Vector> predictions, Tensor<Vector> targets) {
  int N = predictions.data.length;
  double sumSquaredError = 0.0;
  for (int i = 0; i < N; i = i + 1) {
    double error = predictions.data[i] - targets.data[i];
    sumSquaredError = sumSquaredError + (error * error);
  }
  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / N);
  out.creator = Node(
    [predictions, targets],
        () {
      for (int i = 0; i < N; i = i + 1) {
        predictions.grad[i] = predictions.grad[i] +
            out.grad[0] * (2.0 * (predictions.data[i] - targets.data[i])) / N;
      }
    },
    opName: 'mse_vector',
    cost: 3 * N,
  );
  return out;
}

Tensor<Vector> relu(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    double val = v.data[i];
    outValue.add(val > 0.0 ? val : 0.0);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[i] * (v.data[i] > 0.0 ? 1.0 : 0.0);
      }
    },
    opName: 'relu_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> sigmoid(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(1.0 / (1.0 + exp(-v.data[i])));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double val = out.data[i];
        v.grad[i] = v.grad[i] + out.grad[i] * (val * (1.0 - val));
      }
    },
    opName: 'sigmoid_vector',
    cost: N,
  );
  return out;
}

Tensor<Scalar> sum(Tensor<Vector> v) {
  int N = v.data.length;
  double total = 0.0;
  for (int i = 0; i < N; i = i + 1) {
    total = total + v.data[i];
  }
  Tensor<Scalar> out = Tensor<Scalar>(total);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[0];
      }
    },
    opName: 'sum_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorTanh(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    double x = v.data[i];
    double e2x = exp(2.0 * x);
    if (e2x.isInfinite) {
      outValue.add(1.0);
    } else {
      outValue.add((e2x - 1.0) / (e2x + 1.0));
    }
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double val = out.data[i];
        v.grad[i] = v.grad[i] + out.grad[i] * (1.0 - (val * val));
      }
    },
    opName: 'tanh_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorExp(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(exp(v.data[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[i] * out.data[i];
      }
    },
    opName: 'exp_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorLog(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(log(v.data[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[i] * (1.0 / v.data[i]);
      }
    },
    opName: 'log_vector',
    cost: N,
  );
  return out;
}

Tensor<Vector> softplus(Tensor<Vector> v) {
  return vectorLog(addScalar(vectorExp(v), 1.0));
}

Tensor<Vector> avgPool1d(Tensor<Vector> input, int poolSize, int stride) {
  int inputSize = input.shape[0];
  int outputSize = (inputSize - poolSize) ~/ stride + 1;

  Vector outputValue = [];

  for (int i = 0; i < outputSize; i = i + 1) {
    double sum = 0.0;
    for (int p = 0; p < poolSize; p = p + 1) {
      sum = sum + input.data[i * stride + p];
    }
    outputValue.add(sum / poolSize);
  }

  Tensor<Vector> out = Tensor<Vector>(outputValue);

  out.creator = Node(
    [input],
        () {
      double gradDist = 1.0 / poolSize;
      for (int i = 0; i < outputSize; i = i + 1) {
        for (int p = 0; p < poolSize; p = p + 1) {
          int inIdx = i * stride + p;
          input.grad[inIdx] = input.grad[inIdx] + out.grad[i] * gradDist;
        }
      }
    },
    opName: 'avg_pool_1d',
    cost: outputSize * poolSize,
  );

  return out;
}

// ─────────────────────────────────────────────────────── //
// MATRIX (2D) OPERATIONS
// ─────────────────────────────────────────────────────── //

Tensor<Matrix> addMatrix(Tensor<Matrix> a, Tensor<Matrix> b) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  Matrix aMat = a.value;
  Matrix bMat = b.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(aMat[i][j] + bMat[i][j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [a, b],
        () {
      int length = a.data.length;
      for (int i = 0; i < length; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i];
        b.grad[i] = b.grad[i] + out.grad[i];
      }
    },
    opName: 'add_matrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> addMatrixAndVector(Tensor<Matrix> m, Tensor<Vector> v) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;
  Vector vVec = v.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(mMat[i][j] + vVec[j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m, v],
        () {
      for (int i = 0; i < numRows; i = i + 1) {
        for (int j = 0; j < numCols; j = j + 1) {
          int idx = i * numCols + j;
          m.grad[idx] = m.grad[idx] + out.grad[idx];
        }
      }
      for (int j = 0; j < numCols; j = j + 1) {
        for (int i = 0; i < numRows; i = i + 1) {
          int idx = i * numCols + j;
          v.grad[j] = v.grad[j] + out.grad[idx];
        }
      }
    },
    opName: 'addMatrixAndVector',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> addScalarToMatrix(Tensor<Matrix> m, Tensor<Scalar> s) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;
  double sVal = s.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(mMat[i][j] + sVal);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m, s],
        () {
      double sGradSum = 0.0;
      for (int i = 0; i < numRows; i = i + 1) {
        for (int j = 0; j < numCols; j = j + 1) {
          int idx = i * numCols + j;
          m.grad[idx] = m.grad[idx] + out.grad[idx];
          sGradSum = sGradSum + out.grad[idx];
        }
      }
      s.grad[0] = s.grad[0] + sGradSum;
    },
    opName: 'addScalarToMatrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> concatenateMatricesByColumn(List<Tensor<Matrix>> matrices) {
  int numRows = matrices[0].shape[0];

  List<Matrix> cachedMatrices = [];
  for (int k = 0; k < matrices.length; k = k + 1) {
    cachedMatrices.add(matrices[k].value);
  }

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector newRow = [];
    for (int k = 0; k < matrices.length; k = k + 1) {
      Matrix mMat = cachedMatrices[k];
      int mCols = matrices[k].shape[1];
      for (int j = 0; j < mCols; j = j + 1) {
        newRow.add(mMat[i][j]);
      }
    }
    outValue.add(newRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
      matrices,
          () {
        int currentCol = 0;
        int outCols = out.shape[1];
        for (int k = 0; k < matrices.length; k = k + 1) {
          Tensor<Matrix> m = matrices[k];
          int numCols = m.shape[1];

          for (int r = 0; r < numRows; r = r + 1) {
            for (int c = 0; c < numCols; c = c + 1) {
              int mIdx = r * numCols + c;
              int outIdx = r * outCols + (currentCol + c);
              m.grad[mIdx] = m.grad[mIdx] + out.grad[outIdx];
            }
          }
          currentCol = currentCol + numCols;
        }
      },
      opName: 'concat_matrix_col'
  );
  return out;
}

Tensor<Matrix> elementWiseMultiplyMatrix(Tensor<Matrix> a, Tensor<Matrix> b) {
  int numRows = a.shape[0];
  int numCols = a.shape[1];

  Matrix aMat = a.value;
  Matrix bMat = b.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(aMat[i][j] * bMat[i][j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [a, b],
        () {
      int length = a.data.length;
      for (int i = 0; i < length; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i] * b.data[i];
        b.grad[i] = b.grad[i] + out.grad[i] * a.data[i];
      }
    },
    opName: 'multiply_matrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> conv2d(
    Tensor<Matrix> input,
    Tensor<Matrix> kernel, {
      String padding = 'valid',
    }) {
  Matrix inputMatrix = input.value;
  Matrix kernelMatrix = kernel.value;

  int padSize = 0;
  int originalInputHeight = input.shape[0];
  int originalInputWidth = input.shape[1];
  int kernelHeight = kernel.shape[0];
  int kernelWidth = kernel.shape[1];

  if (padding == 'same') {
    padSize = (kernelHeight - 1) ~/ 2;
    inputMatrix = padMatrix(inputMatrix, padSize);
  }

  int inputHeight = inputMatrix.length;
  int inputWidth = inputMatrix[0].length;
  int outputHeight = inputHeight - kernelHeight + 1;
  int outputWidth = inputWidth - kernelWidth + 1;

  Matrix outputValue = [];
  for (int i = 0; i < outputHeight; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < outputWidth; j = j + 1) {
      row.add(0.0);
    }
    outputValue.add(row);
  }

  for (int y = 0; y < outputHeight; y = y + 1) {
    for (int x = 0; x < outputWidth; x = x + 1) {
      double sum = 0.0;
      for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
        for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
          sum = sum + inputMatrix[y + ky][x + kx] * kernelMatrix[ky][kx];
        }
      }
      outputValue[y][x] = sum;
    }
  }

  Tensor<Matrix> out = Tensor<Matrix>(outputValue);
  int cost = outputHeight * outputWidth * 2 * kernelHeight * kernelWidth;

  out.creator = Node(
    [input, kernel],
        () {
      for (int y = 0; y < outputHeight; y = y + 1) {
        for (int x = 0; x < outputWidth; x = x + 1) {
          int outIdx = y * outputWidth + x;

          for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
            for (int kx = 0; kx < kernelWidth; kx = kx + 1) {

              if (padding == 'same' &&
                  (y + ky < padSize ||
                      y + ky >= originalInputHeight + padSize ||
                      x + kx < padSize ||
                      x + kx >= originalInputWidth + padSize)) {
                continue;
              }

              int inputGradY = (padding == 'same') ? y + ky - padSize : y + ky;
              int inputGradX = (padding == 'same') ? x + kx - padSize : x + kx;

              int inIdx = inputGradY * originalInputWidth + inputGradX;
              int kIdx = ky * kernelWidth + kx;

              input.grad[inIdx] = input.grad[inIdx] + kernelMatrix[ky][kx] * out.grad[outIdx];
              kernel.grad[kIdx] = kernel.grad[kIdx] + inputMatrix[y + ky][x + kx] * out.grad[outIdx];
            }
          }
        }
      }
    },
    opName: 'conv2d',
    extraParams: {
      'padding': padding,
    },
    cost: cost,
  );
  return out;
}

Tensor<Matrix> matMul(Tensor<Matrix> a, Tensor<Matrix> b) {
  int M = a.shape[0];
  int N = a.shape[1];
  int P = b.shape[1];

  Matrix aMat = a.value;
  Matrix bMat = b.value;

  Matrix bT = [];
  for (int i = 0; i < P; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < N; j = j + 1) {
      row.add(bMat[j][i]);
    }
    bT.add(row);
  }

  Matrix outValue = [];
  for (int i = 0; i < M; i = i + 1) {
    Vector rowA = aMat[i];
    Vector outRow = [];
    for (int j = 0; j < P; j = j + 1) {
      Vector rowBT = bT[j];
      double sum = 0.0;
      for (int k = 0; k < N; k = k + 1) {
        sum = sum + rowA[k] * rowBT[k];
      }
      outRow.add(sum);
    }
    outValue.add(outRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = 2 * M * N * P;

  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < M; i = i + 1) {
        for (int k = 0; k < N; k = k + 1) {
          int aIdx = i * N + k;
          double aGradSum = 0.0;
          for (int j = 0; j < P; j = j + 1) {
            int outIdx = i * P + j;
            int bIdx = k * P + j;
            double gradOut = out.grad[outIdx];
            aGradSum = aGradSum + gradOut * b.data[bIdx];
            b.grad[bIdx] = b.grad[bIdx] + a.data[aIdx] * gradOut;
          }
          a.grad[aIdx] = a.grad[aIdx] + aGradSum;
        }
      }
    },
    opName: 'matMul',
    cost: cost,
  );
  return out;
}

Tensor<Vector> matVecMul(Tensor<Matrix> M, Tensor<Vector> v) {
  int numRows = M.shape[0];
  int numCols = M.shape[1];

  Matrix mMat = M.value;
  Vector vVec = v.value;

  Vector outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    double sum = 0.0;
    for (int j = 0; j < numCols; j = j + 1) {
      sum = sum + mMat[i][j] * vVec[j];
    }
    outValue.add(sum);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [M, v],
        () {
      for (int i = 0; i < numRows; i = i + 1) {
        double outGrad = out.grad[i];
        for (int j = 0; j < numCols; j = j + 1) {
          int mIdx = i * numCols + j;
          M.grad[mIdx] = M.grad[mIdx] + outGrad * v.data[j];
          v.grad[j] = v.grad[j] + M.data[mIdx] * outGrad;
        }
      }
    },
    opName: 'matVecMul',
    cost: 2 * numRows * numCols,
  );
  return out;
}

Tensor<Scalar> mseMatrix(Tensor<Matrix> predictions, Tensor<Matrix> targets) {
  int length = predictions.data.length;
  double sumSquaredError = 0.0;

  for (int i = 0; i < length; i = i + 1) {
    double error = predictions.data[i] - targets.data[i];
    sumSquaredError = sumSquaredError + (error * error);
  }

  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / length);

  out.creator = Node(
    [predictions, targets],
        () {
      double factor = 2.0 / length;
      double outGrad = out.grad[0];
      for (int i = 0; i < length; i = i + 1) {
        predictions.grad[i] = predictions.grad[i] +
            outGrad * factor * (predictions.data[i] - targets.data[i]);
      }
    },
    opName: 'mse_matrix',
    cost: 3 * length,
  );
  return out;
}

Tensor<Matrix> reluMatrix(Tensor<Matrix> m) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double val = mMat[i][j];
      row.add(val > 0.0 ? val : 0.0);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        m.grad[i] = m.grad[i] + out.grad[i] * (m.data[i] > 0.0 ? 1.0 : 0.0);
      }
    },
    opName: 'relu_matrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> reshapeVectorToMatrix(Tensor<Vector> v, int numRows, int numCols) {
  Vector vVec = v.value;
  Matrix outValue = [];
  int index = 0;

  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(vVec[index]);
      index = index + 1;
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [v],
        () {
      int length = v.data.length;
      for (int i = 0; i < length; i = i + 1) {
        v.grad[i] = v.grad[i] + out.grad[i];
      }
    },
    opName: 'reshape',
    extraParams: {'numRows': numRows, 'numCols': numCols},
    cost: 0,
  );
  return out;
}

Tensor<Matrix> scaleMatrix(Tensor<Matrix> m, double s) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(mMat[i][j] * s);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        m.grad[i] = m.grad[i] + out.grad[i] * s;
      }
    },
    opName: 'scale_matrix',
    cost: numRows * numCols,
    extraParams: {'s': s},
  );
  return out;
}

Tensor<Vector> selectRow(Tensor<Matrix> m, int rowIndex) {
  int numCols = m.shape[1];

  Matrix mMat = m.value;

  Vector outValue = [];
  for (int i = 0; i < numCols; i = i + 1) {
    outValue.add(mMat[rowIndex][i]);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numCols; i = i + 1) {
        int mIdx = rowIndex * numCols + i;
        m.grad[mIdx] = m.grad[mIdx] + out.grad[i];
      }
    },
    opName: 'selectRow',
    extraParams: {'rowIndex': rowIndex},
    cost: 0,
  );
  return out;
}

Tensor<Matrix> sigmoidMatrix(Tensor<Matrix> m) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      row.add(1.0 / (1.0 + exp(-mMat[i][j])));
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double val = out.data[i];
        m.grad[i] = m.grad[i] + out.grad[i] * (val * (1.0 - val));
      }
    },
    opName: 'sigmoid_matrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Scalar> sumMatrix(Tensor<Matrix> m) {
  int length = m.data.length;
  double total = 0.0;

  for (int i = 0; i < length; i = i + 1) {
    total = total + m.data[i];
  }

  Tensor<Scalar> out = Tensor<Scalar>(total);

  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < length; i = i + 1) {
        m.grad[i] = m.grad[i] + out.grad[0];
      }
    },
    opName: 'sum_matrix',
    cost: length,
  );
  return out;
}

Tensor<Matrix> tanhMatrix(Tensor<Matrix> m) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];

  Matrix mMat = m.value;

  Matrix outValue = [];
  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double x = mMat[i][j];
      double e2x = exp(2.0 * x);
      if (e2x.isInfinite) {
        row.add(1.0);
      } else {
        row.add((e2x - 1.0) / (e2x + 1.0));
      }
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double val = out.data[i];
        m.grad[i] = m.grad[i] + out.grad[i] * (1.0 - (val * val));
      }
    },
    opName: 'tanh_matrix',
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> transpose(Tensor<Matrix> a) {
  int M = a.shape[0];
  int N = a.shape[1];

  Matrix aMat = a.value;

  Matrix outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < M; j = j + 1) {
      row.add(aMat[j][i]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [a],
        () {
      for (int i = 0; i < N; i = i + 1) {
        for (int j = 0; j < M; j = j + 1) {
          int aIdx = j * N + i;
          int outIdx = i * M + j;
          a.grad[aIdx] = a.grad[aIdx] + out.grad[outIdx];
        }
      }
    },
    opName: 'transpose',
    cost: 0,
  );
  return out;
}

Tensor<Matrix> softmaxMatrix(Tensor<Matrix> m) {
  Matrix inputMatrix = m.value;
  int numRows = inputMatrix.length;
  int numCols = 0;
  if (numRows > 0) {
    numCols = inputMatrix[0].length;
  }

  Matrix outValue = [];

  for (int r = 0; r < numRows; r = r + 1) {
    Vector row = inputMatrix[r];

    double maxVal = -double.infinity;
    for (int c = 0; c < numCols; c = c + 1) {
      if (row[c] > maxVal) {
        maxVal = row[c];
      }
    }

    double sumExps = 0.0;
    Vector exps = [];
    for (int c = 0; c < numCols; c = c + 1) {
      double expVal = exp(row[c] - maxVal);
      exps.add(expVal);
      sumExps = sumExps + expVal;
    }

    Vector softmaxRow = [];
    for (int c = 0; c < numCols; c = c + 1) {
      softmaxRow.add(exps[c] / sumExps);
    }
    outValue.add(softmaxRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
      [m],
          () {
        for (int r = 0; r < numRows; r = r + 1) {
          double dotProduct = 0.0;
          for (int c = 0; c < numCols; c = c + 1) {
            int flatIndex = r * numCols + c;
            double y_c = out.data[flatIndex];
            double dy_c = out.grad[flatIndex];
            dotProduct = dotProduct + (dy_c * y_c);
          }

          for (int c = 0; c < numCols; c = c + 1) {
            int flatIndex = r * numCols + c;
            double y_c = out.data[flatIndex];
            double dy_c = out.grad[flatIndex];
            m.grad[flatIndex] = m.grad[flatIndex] + (y_c * (dy_c - dotProduct));
          }
        }
      },
      opName: 'softmax_matrix',
      cost: numRows * numCols * 2
  );

  return out;
}

// ─────────────────────────────────────────────────────── //
// 3D TENSOR OPERATIONS
// ─────────────────────────────────────────────────────── //

Tensor<Tensor3D> add3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  Tensor3D aVal = a.value;
  Tensor3D bVal = b.value;

  Tensor3D outValue = [];

  for (int d = 0; d < depth; d = d + 1) {
    Matrix matrix = [];
    for (int h = 0; h < height; h = h + 1) {
      Vector row = [];
      for (int w = 0; w < width; w = w + 1) {
        row.add(aVal[d][h][w] + bVal[d][h][w]);
      }
      matrix.add(row);
    }
    outValue.add(matrix);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
  out.creator = Node(
    [a, b],
        () {
      int length = a.data.length;
      for (int i = 0; i < length; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i];
        b.grad[i] = b.grad[i] + out.grad[i];
      }
    },
    opName: 'add_3d',
    cost: depth * height * width,
  );
  return out;
}

Tensor<Tensor3D> elementWiseMultiply3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int depth = a.shape[0];
  int height = a.shape[1];
  int width = a.shape[2];

  Tensor3D aVal = a.value;
  Tensor3D bVal = b.value;

  Tensor3D outValue = [];

  for (int d = 0; d < depth; d = d + 1) {
    Matrix matrix = [];
    for (int h = 0; h < height; h = h + 1) {
      Vector row = [];
      for (int w = 0; w < width; w = w + 1) {
        row.add(aVal[d][h][w] * bVal[d][h][w]);
      }
      matrix.add(row);
    }
    outValue.add(matrix);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
  out.creator = Node(
    [a, b],
        () {
      int length = a.data.length;
      for (int i = 0; i < length; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i] * b.data[i];
        b.grad[i] = b.grad[i] + out.grad[i] * a.data[i];
      }
    },
    opName: 'multiply_3d',
    cost: depth * height * width,
  );
  return out;
}

Tensor<Tensor3D> concatenate3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int aDepth = a.shape[0];
  int bDepth = b.shape[0];

  Tensor3D aVal = a.value;
  Tensor3D bVal = b.value;

  Tensor3D outValue = [];
  for (int d = 0; d < aDepth; d = d + 1) {
    outValue.add(aVal[d]);
  }
  for (int d = 0; d < bDepth; d = d + 1) {
    outValue.add(bVal[d]);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);

  out.creator = Node(
    [a, b],
        () {
      int aLen = a.data.length;
      for (int i = 0; i < aLen; i = i + 1) {
        a.grad[i] = a.grad[i] + out.grad[i];
      }

      int bLen = b.data.length;
      for (int i = 0; i < bLen; i = i + 1) {
        b.grad[i] = b.grad[i] + out.grad[aLen + i];
      }
    },
    opName: 'concat_3d',
    cost: 0,
  );
  return out;
}



// ─────────────────────────────────────────────────────── //
// AVERAGE POOLING OPERATIONS
// ─────────────────────────────────────────────────────── //


Tensor<Matrix> avgPool2d(Tensor<Matrix> input, int poolSize, int stride) {
  int inputHeight = input.shape[0];
  int inputWidth = input.shape[1];

  int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
  int outputWidth = (inputWidth - poolSize) ~/ stride + 1;

  Matrix outputValue = [];
  double numElements = (poolSize * poolSize).toDouble();

  for (int y = 0; y < outputHeight; y = y + 1) {
    Vector row = [];
    for (int x = 0; x < outputWidth; x = x + 1) {
      double sum = 0.0;
      for (int py = 0; py < poolSize; py = py + 1) {
        int inY = y * stride + py;
        for (int px = 0; px < poolSize; px = px + 1) {
          int inX = x * stride + px;
          sum = sum + input.data[inY * inputWidth + inX];
        }
      }
      row.add(sum / numElements);
    }
    outputValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outputValue);

  out.creator = Node(
    [input],
        () {
      double gradDist = 1.0 / numElements;
      for (int y = 0; y < outputHeight; y = y + 1) {
        for (int x = 0; x < outputWidth; x = x + 1) {
          int outIdx = y * outputWidth + x;
          for (int py = 0; py < poolSize; py = py + 1) {
            for (int px = 0; px < poolSize; px = px + 1) {
              int inY = y * stride + py;
              int inX = x * stride + px;
              int inIdx = inY * inputWidth + inX;
              input.grad[inIdx] = input.grad[inIdx] + out.grad[outIdx] * gradDist;
            }
          }
        }
      }
    },
    opName: 'avg_pool_2d',
    cost: outputHeight * outputWidth * poolSize * poolSize,
  );

  return out;
}

Tensor<Vector> globalAveragePooling(Tensor<Matrix> input) {
  int sequenceLength = input.shape[0];
  int dModel = input.shape[1];

  Vector averagedVector = [];
  for (int c = 0; c < dModel; c = c + 1) {
    averagedVector.add(0.0);
  }

  for (int r = 0; r < sequenceLength; r = r + 1) {
    for (int c = 0; c < dModel; c = c + 1) {
      averagedVector[c] = averagedVector[c] + input.data[r * dModel + c];
    }
  }

  for (int c = 0; c < dModel; c = c + 1) {
    averagedVector[c] = averagedVector[c] / sequenceLength;
  }

  Tensor<Vector> out = Tensor<Vector>(averagedVector);

  out.creator = Node(
    [input],
        () {
      for (int r = 0; r < sequenceLength; r = r + 1) {
        for (int c = 0; c < dModel; c = c + 1) {
          int inIdx = r * dModel + c;
          input.grad[inIdx] = input.grad[inIdx] + out.grad[c] / sequenceLength;
        }
      }
    },
    opName: 'global_avg_pool',
    cost: sequenceLength * dModel,
  );

  return out;
}

Tensor<Vector> batchNorm1dMath(
    Tensor<Vector> x,
    Tensor<Vector> gamma,
    Tensor<Vector> beta,
    Vector runningMean,
    Vector runningVariance,
    int numFeatures,
    bool isTraining,
    double momentum,
    double epsilon,
    ) {
  Vector xHat = [];
  Vector currentMean = [];
  Vector currentVariance = [];

  if (isTraining) {
    for (int i = 0; i < numFeatures; i = i + 1) {
      currentMean.add(x.data[i]);
      currentVariance.add(0.0);
    }

    for (int i = 0; i < numFeatures; i = i + 1) {
      runningMean[i] = momentum * runningMean[i] + (1.0 - momentum) * currentMean[i];
      runningVariance[i] = momentum * runningVariance[i] + (1.0 - momentum) * currentVariance[i];
    }
  } else {
    for (int i = 0; i < numFeatures; i = i + 1) {
      currentMean.add(runningMean[i]);
      currentVariance.add(runningVariance[i]);
    }
  }

  Vector varianceToUse = isTraining ? runningVariance : currentVariance;
  Vector meanToUse = isTraining ? runningMean : currentMean;

  for (int i = 0; i < numFeatures; i = i + 1) {
    xHat.add((x.data[i] - meanToUse[i]) / sqrt(varianceToUse[i] + epsilon));
  }

  Vector outValue = [];
  for (int i = 0; i < numFeatures; i = i + 1) {
    outValue.add(gamma.data[i] * xHat[i] + beta.data[i]);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [x, gamma, beta],
        () {
      for (int i = 0; i < numFeatures; i = i + 1) {
        double invStd = 1.0 / sqrt(varianceToUse[i] + epsilon);
        gamma.grad[i] = gamma.grad[i] + out.grad[i] * xHat[i];
        beta.grad[i] = beta.grad[i] + out.grad[i];
        x.grad[i] = x.grad[i] + out.grad[i] * gamma.data[i] * invStd;
      }
    },
    opName: 'batch_norm_1d',
    cost: numFeatures * 4,
  );

  return out;
}

Tensor<Tensor3D> batchNorm2dMath(
    Tensor<Tensor3D> x,
    Tensor<Vector> gamma,
    Tensor<Vector> beta,
    Vector runningMean,
    Vector runningVariance,
    int numChannels,
    bool isTraining,
    double momentum,
    double epsilon,
    ) {
  int height = x.shape[1];
  int width = x.shape[2];
  int planeSize = height * width;
  double numElements = planeSize.toDouble();

  Vector currentMean = [];
  Vector currentVariance = [];
  for (int c = 0; c < numChannels; c = c + 1) {
    currentMean.add(0.0);
    currentVariance.add(0.0);
  }

  Vector meanToUse = [];
  Vector varianceToUse = [];

  if (isTraining) {
    for (int c = 0; c < numChannels; c = c + 1) {
      double sum = 0.0;
      int cOffset = c * planeSize;
      for (int i = 0; i < planeSize; i = i + 1) {
        sum = sum + x.data[cOffset + i];
      }
      currentMean[c] = sum / numElements;

      double varianceSum = 0.0;
      for (int i = 0; i < planeSize; i = i + 1) {
        double diff = x.data[cOffset + i] - currentMean[c];
        varianceSum = varianceSum + (diff * diff);
      }
      currentVariance[c] = varianceSum / numElements;
    }

    for (int c = 0; c < numChannels; c = c + 1) {
      runningMean[c] = momentum * runningMean[c] + (1.0 - momentum) * currentMean[c];
      runningVariance[c] = momentum * runningVariance[c] + (1.0 - momentum) * currentVariance[c];
    }
    meanToUse = currentMean;
    varianceToUse = currentVariance;
  } else {
    meanToUse = runningMean;
    varianceToUse = runningVariance;
  }

  List<double> xHatFlat = [];
  Tensor3D outValue = [];

  for (int c = 0; c < numChannels; c = c + 1) {
    Matrix m = [];
    double mean = meanToUse[c];
    double invStd = 1.0 / sqrt(varianceToUse[c] + epsilon);
    double gammaVal = gamma.data[c];
    double betaVal = beta.data[c];
    int cOffset = c * planeSize;

    for (int h = 0; h < height; h = h + 1) {
      Vector row = [];
      int hOffset = h * width;
      for (int w = 0; w < width; w = w + 1) {
        int flatIdx = cOffset + hOffset + w;
        double val = (x.data[flatIdx] - mean) * invStd;
        xHatFlat.add(val);
        row.add(gammaVal * val + betaVal);
      }
      m.add(row);
    }
    outValue.add(m);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);

  out.creator = Node(
    [x, gamma, beta],
        () {
      for (int c = 0; c < numChannels; c = c + 1) {
        double invStd = 1.0 / sqrt(varianceToUse[c] + epsilon);
        double gVal = gamma.data[c];
        int cOffset = c * planeSize;

        for (int i = 0; i < planeSize; i = i + 1) {
          int flatIdx = cOffset + i;
          double gradOut = out.grad[flatIdx];

          gamma.grad[c] = gamma.grad[c] + gradOut * xHatFlat[flatIdx];
          beta.grad[c] = beta.grad[c] + gradOut;
          x.grad[flatIdx] = x.grad[flatIdx] + gradOut * gVal * invStd;
        }
      }
    },
    opName: 'batch_norm_2d',
    cost: numChannels * height * width * 4,
  );

  return out;
}

Tensor<Tensor3D> stackMatricesTo3D(List<Tensor<Matrix>> matrices) {
  int depth = matrices.length;
  int height = matrices[0].shape[0];
  int width = matrices[0].shape[1];

  Tensor3D outValue = [];
  for (int d = 0; d < depth; d = d + 1) {
    outValue.add(matrices[d].value);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);

  out.creator = Node(
    matrices,
        () {
      for (int d = 0; d < depth; d = d + 1) {
        Tensor<Matrix> m = matrices[d];
        int offset = d * height * width;
        for (int i = 0; i < height * width; i = i + 1) {
          m.grad[i] = m.grad[i] + out.grad[offset + i];
        }
      }
    },
    opName: 'stack_to_3d',
    cost: 0,
  );

  return out;
}

Tensor<Vector> dropoutVectorMath(Tensor<Vector> input, double rate, bool isTraining) {
  if (isTraining == false || rate == 0.0) {
    return input;
  }

  double scale = 1.0 / (1.0 - rate);
  Random random = Random();
  int length = input.data.length;

  Vector outputValue = [];
  List<bool> mask = [];

  for (int i = 0; i < length; i = i + 1) {
    if (random.nextDouble() < rate) {
      outputValue.add(0.0);
      mask.add(false);
    } else {
      outputValue.add(input.data[i] * scale);
      mask.add(true);
    }
  }

  Tensor<Vector> out = Tensor<Vector>(outputValue);

  out.creator = Node(
    [input],
        () {
      for (int i = 0; i < length; i = i + 1) {
        if (mask[i]) {
          input.grad[i] = input.grad[i] + out.grad[i] * scale;
        }
      }
    },
    opName: 'dropout_vector',
    cost: length,
  );

  return out;
}

Tensor<Matrix> dropoutMatrixMath(Tensor<Matrix> input, double rate, bool isTraining) {
  if (isTraining == false || rate == 0.0) {
    return input;
  }

  double scale = 1.0 / (1.0 - rate);
  Random random = Random();
  int rows = input.shape[0];
  int cols = input.shape[1];

  Matrix outputValue = [];
  List<bool> flatMask = [];

  for (int r = 0; r < rows; r = r + 1) {
    Vector row = [];
    for (int c = 0; c < cols; c = c + 1) {
      if (random.nextDouble() < rate) {
        row.add(0.0);
        flatMask.add(false);
      } else {
        row.add(input.data[r * cols + c] * scale);
        flatMask.add(true);
      }
    }
    outputValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outputValue);

  out.creator = Node(
    [input],
        () {
      int length = rows * cols;
      for (int i = 0; i < length; i = i + 1) {
        if (flatMask[i]) {
          input.grad[i] = input.grad[i] + out.grad[i] * scale;
        }
      }
    },
    opName: 'dropout_matrix',
    cost: rows * cols,
  );

  return out;
}

Tensor<Vector> maxPool1d(Tensor<Vector> input, int poolSize, int stride) {
  int inputSize = input.shape[0];
  int outputSize = (inputSize - poolSize) ~/ stride + 1;

  Vector outputValue = [];
  List<int> maxIndices = [];

  for (int i = 0; i < outputSize; i = i + 1) {
    double maxVal = -double.infinity;
    int maxIdx = -1;
    for (int p = 0; p < poolSize; p = p + 1) {
      int inIdx = i * stride + p;
      double val = input.data[inIdx];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = inIdx;
      }
    }
    outputValue.add(maxVal);
    maxIndices.add(maxIdx);
  }

  Tensor<Vector> out = Tensor<Vector>(outputValue);

  out.creator = Node(
    [input],
        () {
      int outLength = out.data.length;
      for (int i = 0; i < outLength; i = i + 1) {
        int mIdx = maxIndices[i];
        input.grad[mIdx] = input.grad[mIdx] + out.grad[i];
      }
    },
    opName: 'max_pool_1d',
    cost: outputSize * poolSize,
  );

  return out;
}

Tensor<Matrix> maxPool2d(Tensor<Matrix> input, int poolSize, int stride) {
  int inputHeight = input.shape[0];
  int inputWidth = input.shape[1];

  int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
  int outputWidth = (inputWidth - poolSize) ~/ stride + 1;

  Matrix outputValue = [];
  List<int> maxIndices = [];

  for (int y = 0; y < outputHeight; y = y + 1) {
    Vector row = [];
    for (int x = 0; x < outputWidth; x = x + 1) {
      double maxVal = -double.infinity;
      int maxIdx = -1;
      for (int py = 0; py < poolSize; py = py + 1) {
        int inY = y * stride + py;
        for (int px = 0; px < poolSize; px = px + 1) {
          int inX = x * stride + px;
          int flatInIdx = inY * inputWidth + inX;
          double val = input.data[flatInIdx];
          if (val > maxVal) {
            maxVal = val;
            maxIdx = flatInIdx;
          }
        }
      }
      row.add(maxVal);
      maxIndices.add(maxIdx);
    }
    outputValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outputValue);

  out.creator = Node(
    [input],
        () {
      int outLength = out.data.length;
      for (int i = 0; i < outLength; i = i + 1) {
        int mIdx = maxIndices[i];
        input.grad[mIdx] = input.grad[mIdx] + out.grad[i];
      }
    },
    opName: 'max_pool_2d',
    cost: outputHeight * outputWidth * poolSize * poolSize,
  );

  return out;
}


Tensor<Vector> softmaxVector(Tensor<Vector> v) {
  int N = v.data.length;
  double maxVal = -double.infinity;

  for (int i = 0; i < N; i = i + 1) {
    if (v.data[i] > maxVal) {
      maxVal = v.data[i];
    }
  }

  double sumExps = 0.0;
  Vector exps = [];
  for (int i = 0; i < N; i = i + 1) {
    double expVal = exp(v.data[i] - maxVal);
    exps.add(expVal);
    sumExps = sumExps + expVal;
  }

  Vector outValue = [];
  for (int i = 0; i < N; i = i + 1) {
    outValue.add(exps[i] / sumExps);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [v],
        () {
      double dotProduct = 0.0;
      for (int i = 0; i < N; i = i + 1) {
        dotProduct = dotProduct + (out.grad[i] * out.data[i]);
      }

      for (int i = 0; i < N; i = i + 1) {
        v.grad[i] = v.grad[i] + out.data[i] * (out.grad[i] - dotProduct);
      }
    },
    opName: 'softmax_vector',
    cost: N * 2,
  );

  return out;
}

Tensor<Vector> swishVector(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];

  for (int i = 0; i < N; i = i + 1) {
    double x = v.data[i];
    double sigVal = 1.0 / (1.0 + exp(-x));
    outValue.add(x * sigVal);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double x = v.data[i];
        double sigVal = 1.0 / (1.0 + exp(-x));
        double derivative = sigVal * (1.0 + x * (1.0 - sigVal));
        v.grad[i] = v.grad[i] + out.grad[i] * derivative;
      }
    },
    opName: 'swish_vector',
    cost: N * 2,
  );

  return out;
}

Tensor<Matrix> swishMatrix(Tensor<Matrix> m) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];
  Matrix mMat = m.value;

  Matrix outValue = [];

  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double x = mMat[i][j];
      double sigVal = 1.0 / (1.0 + exp(-x));
      row.add(x * sigVal);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double x = m.data[i];
        double sigVal = 1.0 / (1.0 + exp(-x));
        double derivative = sigVal * (1.0 + x * (1.0 - sigVal));
        m.grad[i] = m.grad[i] + out.grad[i] * derivative;
      }
    },
    opName: 'swish_matrix',
    cost: numRows * numCols * 2,
  );

  return out;
}


// ─────────────────────────────────────────────────────── //
// ELU (Exponential Linear Unit)
// ─────────────────────────────────────────────────────── //

Tensor<Vector> eluVector(Tensor<Vector> v, double alpha) {
  int N = v.data.length;
  Vector outValue = [];

  for (int i = 0; i < N; i = i + 1) {
    double x = v.data[i];
    if (x > 0.0) {
      outValue.add(x);
    } else {
      outValue.add(alpha * (exp(x) - 1.0));
    }
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double x = v.data[i];
        double grad = x > 0.0 ? 1.0 : out.data[i] + alpha;
        v.grad[i] = v.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'elu_vector',
    cost: N,
  );

  return out;
}

Tensor<Matrix> eluMatrix(Tensor<Matrix> m, double alpha) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];
  Matrix mMat = m.value;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double x = mMat[i][j];
      if (x > 0.0) {
        row.add(x);
      } else {
        row.add(alpha * (exp(x) - 1.0));
      }
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double x = m.data[i];
        double grad = x > 0.0 ? 1.0 : out.data[i] + alpha;
        m.grad[i] = m.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'elu_matrix',
    cost: numRows * numCols,
  );

  return out;
}

// ─────────────────────────────────────────────────────── //
// LEAKY RELU
// ─────────────────────────────────────────────────────── //

Tensor<Vector> leakyReluVector(Tensor<Vector> v, double alpha) {
  int N = v.data.length;
  Vector outValue = [];

  for (int i = 0; i < N; i = i + 1) {
    double x = v.data[i];
    if (x > 0.0) {
      outValue.add(x);
    } else {
      outValue.add(alpha * x);
    }
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double x = v.data[i];
        double grad = x > 0.0 ? 1.0 : alpha;
        v.grad[i] = v.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'leaky_relu_vector',
    cost: N,
  );

  return out;
}

Tensor<Matrix> leakyReluMatrix(Tensor<Matrix> m, double alpha) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];
  Matrix mMat = m.value;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double x = mMat[i][j];
      if (x > 0.0) {
        row.add(x);
      } else {
        row.add(alpha * x);
      }
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double x = m.data[i];
        double grad = x > 0.0 ? 1.0 : alpha;
        m.grad[i] = m.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'leaky_relu_matrix',
    cost: numRows * numCols,
  );

  return out;
}

// ─────────────────────────────────────────────────────── //
// MISH
// ─────────────────────────────────────────────────────── //

Tensor<Vector> mishVector(Tensor<Vector> v) {
  int N = v.data.length;
  Vector outValue = [];

  for (int i = 0; i < N; i = i + 1) {
    double x = v.data[i];
    double sp = log(1.0 + exp(x));
    double e2sp = exp(2.0 * sp);
    double t = e2sp.isInfinite ? 1.0 : (e2sp - 1.0) / (e2sp + 1.0);
    outValue.add(x * t);
  }

  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i = i + 1) {
        double x = v.data[i];
        double sp = log(1.0 + exp(x));
        double e2sp = exp(2.0 * sp);
        double t = e2sp.isInfinite ? 1.0 : (e2sp - 1.0) / (e2sp + 1.0);
        double s = 1.0 / (1.0 + exp(-x));
        double grad = t + x * s * (1.0 - t * t);
        v.grad[i] = v.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'mish_vector',
    cost: N * 3,
  );

  return out;
}

Tensor<Matrix> mishMatrix(Tensor<Matrix> m) {
  int numRows = m.shape[0];
  int numCols = m.shape[1];
  Matrix mMat = m.value;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < numCols; j = j + 1) {
      double x = mMat[i][j];
      double sp = log(1.0 + exp(x));
      double e2sp = exp(2.0 * sp);
      double t = e2sp.isInfinite ? 1.0 : (e2sp - 1.0) / (e2sp + 1.0);
      row.add(x * t);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(
    [m],
        () {
      int length = m.data.length;
      for (int i = 0; i < length; i = i + 1) {
        double x = m.data[i];
        double sp = log(1.0 + exp(x));
        double e2sp = exp(2.0 * sp);
        double t = e2sp.isInfinite ? 1.0 : (e2sp - 1.0) / (e2sp + 1.0);
        double s = 1.0 / (1.0 + exp(-x));
        double grad = t + x * s * (1.0 - t * t);
        m.grad[i] = m.grad[i] + out.grad[i] * grad;
      }
    },
    opName: 'mish_matrix',
    cost: numRows * numCols * 3,
  );

  return out;
}