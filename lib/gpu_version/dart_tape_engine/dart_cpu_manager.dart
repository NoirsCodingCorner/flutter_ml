import 'dart:ffi';
import 'dart:typed_data';
import 'dart:math';

class DartCpuElement {
  Float32List data;
  List<int> sizes;
  int totalElements;

  DartCpuElement(this.data, this.sizes, this.totalElements);
}

class DartCpuManager {
  Map<String, DartCpuElement> memory = <String, DartCpuElement>{};

  // ─────────────────────────────────────────────────────── //
  // Memory Management
  // ─────────────────────────────────────────────────────── //

  void allocate(String name, List<int> sizes) {
    int totalElements = 1;
    List<int> shapeCopy = <int>[];
    for (int i = 0; i < sizes.length; i = i + 1) {
      totalElements = totalElements * sizes[i];
      shapeCopy.add(sizes[i]);
    }

    Float32List buffer = Float32List(totalElements);
    memory[name] = DartCpuElement(buffer, shapeCopy, totalElements);
  }

  bool exists(String name) {
    return memory.containsKey(name);
  }

  void free(String name) {
    memory.remove(name);
  }

  void clear() {
    memory.clear();
  }

  DartCpuElement getElement(String name) {
    return memory[name]!;
  }

  void copyHostToDevice(String name, Pointer<Float> hostPtr) {
    DartCpuElement elem = memory[name]!;
    Float32List src = hostPtr.asTypedList(elem.totalElements);
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      elem.data[i] = src[i];
    }
  }

  void retrieveInto(String name, Pointer<Float> destPtr) {
    DartCpuElement elem = memory[name]!;
    Float32List dest = destPtr.asTypedList(elem.totalElements);
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      dest[i] = elem.data[i];
    }
  }

  // ─────────────────────────────────────────────────────── //
  // Operation Stubs (Math Implementations Go Here)
  // ─────────────────────────────────────────────────────── //

  void zeroGrad(String name) {
    DartCpuElement elem = memory[name]!;
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      elem.data[i] = 0.0;
    }
  }

  void copy(String nameSrc, String nameDest) {
    DartCpuElement src = memory[nameSrc]!;
    DartCpuElement dest = memory[nameDest]!;
    for (int i = 0; i < src.totalElements; i = i + 1) {
      dest.data[i] = src.data[i];
    }
  }

  void fill(String name, double value) {
    DartCpuElement elem = memory[name]!;
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      elem.data[i] = value;
    }
  }

  void initRandomUniform(String name, double scale, int seed) {
    DartCpuElement elem = memory[name]!;
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      int state = (i ^ seed) & 0xFFFFFFFF;
      state = (state ^ (state >> 16)) & 0xFFFFFFFF;
      state = (state * 0x7feb352d) & 0xFFFFFFFF;
      state = (state ^ (state >> 15)) & 0xFFFFFFFF;
      state = (state * 0x846ca68b) & 0xFFFFFFFF;
      state = (state ^ (state >> 16)) & 0xFFFFFFFF;

      double randVal = (state / 4294967295.0) * 2.0 - 1.0;
      elem.data[i] = randVal * scale;
    }
  }

  void addScalar(String nameIn, String nameOut, double scalar) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = inElem.data[i] + scalar;
    }
  }

  void pad2d(
      String nameIn, String nameOut, int padT, int padB, int padL, int padR) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int inH = inElem.sizes[0];
    int inW = inElem.sizes[1];
    int outH = outElem.sizes[0];
    int outW = outElem.sizes[1];

    for (int outR = 0; outR < outH; outR = outR + 1) {
      int inR = outR - padT;
      for (int outC = 0; outC < outW; outC = outC + 1) {
        int inC = outC - padL;
        int outIdx = outR * outW + outC;

        if (inR >= 0 && inR < inH && inC >= 0 && inC < inW) {
          outElem.data[outIdx] = inElem.data[inR * inW + inC];
        } else {
          outElem.data[outIdx] = 0.0;
        }
      }
    }
  }

  void pad2dBackward(String nameGradOut, String nameGradIn, int padT, int padB,
      int padL, int padR) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int inH = gradInElem.sizes[0];
    int inW = gradInElem.sizes[1];
    int outW = gradOutElem.sizes[1];

    for (int inR = 0; inR < inH; inR = inR + 1) {
      int outR = inR + padT;
      for (int inC = 0; inC < inW; inC = inC + 1) {
        int outC = inC + padL;
        int inIdx = inR * inW + inC;
        int outIdx = outR * outW + outC;
        gradInElem.data[inIdx] =
            gradInElem.data[inIdx] + gradOutElem.data[outIdx];
      }
    }
  }

  void add(String nameA, String nameB, String nameC) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;
    for (int i = 0; i < c.totalElements; i = i + 1) {
      c.data[i] = a.data[i] + b.data[i];
    }
  }

  void addInto(String nameSrc, String nameDest) {
    DartCpuElement src = memory[nameSrc]!;
    DartCpuElement dest = memory[nameDest]!;
    for (int i = 0; i < src.totalElements; i = i + 1) {
      dest.data[i] = dest.data[i] + src.data[i];
    }
  }

  void subtract(String nameA, String nameB, String nameC) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;
    for (int i = 0; i < a.totalElements; i = i + 1) {
      c.data[i] = a.data[i] - b.data[i];
    }
  }

  void subtractInto(String nameSrc, String nameDest) {
    DartCpuElement src = memory[nameSrc]!;
    DartCpuElement dest = memory[nameDest]!;
    for (int i = 0; i < src.totalElements; i = i + 1) {
      dest.data[i] = dest.data[i] - src.data[i];
    }
  }

  void multiply(String nameA, String nameB, String nameC) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;
    for (int i = 0; i < a.totalElements; i = i + 1) {
      c.data[i] = a.data[i] * b.data[i];
    }
  }

  void multiplyBackward(
      String gradOutName, String otherInputName, String gradInName) {
    DartCpuElement gradOut = memory[gradOutName]!;
    DartCpuElement otherIn = memory[otherInputName]!;
    DartCpuElement gradIn = memory[gradInName]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * otherIn.data[i];
    }
  }

  void divide(String nameA, String nameB, String nameC) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;
    for (int i = 0; i < a.totalElements; i = i + 1) {
      c.data[i] = a.data[i] / b.data[i];
    }
  }

  void divideBackward(String nameA, String nameB, String gradOutName,
      String gradAName, String gradBName) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement gradOut = memory[gradOutName]!;
    DartCpuElement gradA = memory[gradAName]!;
    DartCpuElement gradB = memory[gradBName]!;
    for (int i = 0; i < a.totalElements; i = i + 1) {
      double aVal = a.data[i];
      double bVal = b.data[i];
      double gOut = gradOut.data[i];
      gradA.data[i] = gradA.data[i] + gOut / bVal;
      gradB.data[i] = gradB.data[i] + gOut * (-aVal / (bVal * bVal));
    }
  }

  void expElementwise(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = exp(inElem.data[i]);
    }
  }

  void expBackward(String nameGradOut, String nameOutData, String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement outData = memory[nameOutData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * outData.data[i];
    }
  }

  void logElementwise(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double val = inElem.data[i];
      if (val < 1e-7) {
        val = 1e-7;
      }
      outElem.data[i] = log(val);
    }
  }

  void logBackward(String nameGradOut, String nameInData, String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement inData = memory[nameInData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      double val = inData.data[i];
      if (val < 1e-7) {
        val = 1e-7;
      }
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] / val;
    }
  }

  void absElementwise(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = inElem.data[i].abs();
    }
  }

  void absBackward(String nameGradOut, String nameInData, String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement inData = memory[nameInData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      double val = inData.data[i];
      double sign = 0.0;
      if (val > 0.0) {
        sign = 1.0;
      } else if (val < 0.0) {
        sign = -1.0;
      }
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * sign;
    }
  }

  void sqrtElementwise(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double val = inElem.data[i];
      if (val < 1e-7) {
        val = 1e-7;
      }
      outElem.data[i] = sqrt(val);
    }
  }

  void sqrtBackward(String nameGradOut, String nameOutData, String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement outData = memory[nameOutData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      double val = outData.data[i];
      if (val < 1e-7) {
        val = 1e-7;
      }
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] / (2.0 * val);
    }
  }

  void powElementwise(String nameIn, String nameOut, double exponent) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = pow(inElem.data[i], exponent).toDouble();
    }
  }

  void powBackward(String nameGradOut, String nameInData, String nameGradIn,
      double exponent) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement inData = memory[nameInData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      double deriv = exponent * pow(inData.data[i], exponent - 1.0).toDouble();
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * deriv;
    }
  }

  void clampElementwise(
      String nameIn, String nameOut, double minVal, double maxVal) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double val = inElem.data[i];
      if (val < minVal) {
        val = minVal;
      }
      if (val > maxVal) {
        val = maxVal;
      }
      outElem.data[i] = val;
    }
  }

  void clampBackward(String nameGradOut, String nameInData, String nameGradIn,
      double minVal, double maxVal) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement inData = memory[nameInData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      double val = inData.data[i];
      if (val >= minVal && val <= maxVal) {
        gradIn.data[i] = gradIn.data[i] + gradOut.data[i];
      }
    }
  }

  void matmul(String nameA, String nameB, String nameC, bool transposeA,
      bool transposeB, double alpha, double beta, bool useTensorCores) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;

    int rowsA = a.sizes[0];
    int colsA = a.sizes.length == 1 ? 1 : a.sizes[1];
    int rowsB = b.sizes[0];
    int colsB = b.sizes.length == 1 ? 1 : b.sizes[1];

    int m = transposeA ? colsA : rowsA;
    int k = transposeA ? rowsA : colsA;
    int n = transposeB ? rowsB : colsB;

    for (int i = 0; i < m; i = i + 1) {
      for (int j = 0; j < n; j = j + 1) {
        double sum = 0.0;
        for (int p = 0; p < k; p = p + 1) {
          int aIdx = transposeA ? (p * colsA + i) : (i * colsA + p);
          int bIdx = transposeB ? (j * colsB + p) : (p * colsB + j);
          sum = sum + a.data[aIdx] * b.data[bIdx];
        }
        int cIdx = i * n + j;
        if (beta == 0.0) {
          c.data[cIdx] = alpha * sum;
        } else {
          c.data[cIdx] = alpha * sum + beta * c.data[cIdx];
        }
      }
    }
  }

  void transpose(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int height = inElem.sizes[0];
    int width = inElem.sizes[1];

    for (int r = 0; r < height; r = r + 1) {
      for (int c = 0; c < width; c = c + 1) {
        outElem.data[c * height + r] = inElem.data[r * width + c];
      }
    }
  }

  void broadcastAdd(String nameA, String nameB, String nameC) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameC]!;

    int rows = c.sizes[0];
    int cols = c.sizes[1];

    int bType = 0;
    if (b.totalElements == 1) {
      bType = 0;
    } else {
      int mB = b.sizes.length == 1 ? 1 : b.sizes[0];
      int nB = b.sizes.length == 1 ? b.sizes[0] : b.sizes[1];
      if (mB == 1 && nB == cols) {
        bType = 1;
      } else if ((mB == rows && nB == 1) || (mB == 1 && nB == rows)) {
        bType = 2;
      }
    }

    for (int r = 0; r < rows; r = r + 1) {
      for (int col = 0; col < cols; col = col + 1) {
        double bVal = 0.0;
        if (bType == 0) {
          bVal = b.data[0];
        } else if (bType == 1) {
          bVal = b.data[col];
        } else if (bType == 2) {
          bVal = b.data[r];
        }
        int idx = r * cols + col;
        c.data[idx] = a.data[idx] + bVal;
      }
    }
  }

  void scaleMatrix(String nameIn, String nameOut, double scale) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = inElem.data[i] * scale;
    }
  }

  void scaleMatrixBackward(
      String nameGradOut, String nameGradIn, double scale) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < gradOut.totalElements; i = i + 1) {
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * scale;
    }
  }

  void relu(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double val = inElem.data[i];
      outElem.data[i] = val > 0.0 ? val : 0.0;
    }
  }

  void reluBackward(String nameIn, String nameGradOut, String nameGradIn) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      if (inElem.data[i] > 0.0) {
        gradIn.data[i] = gradIn.data[i] + gradOut.data[i];
      }
    }
  }

  void sigmoid(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      outElem.data[i] = 1.0 / (1.0 + exp(-inElem.data[i]));
    }
  }

  void sigmoidBackward(
      String nameOutData, String nameGradOut, String nameGradIn) {
    DartCpuElement outData = memory[nameOutData]!;
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < outData.totalElements; i = i + 1) {
      double s = outData.data[i];
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * s * (1.0 - s);
    }
  }

  void tanh(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double e2x = exp(2.0 * inElem.data[i]);
      if (e2x.isInfinite) {
        outElem.data[i] = 1.0;
      } else {
        outElem.data[i] = (e2x - 1.0) / (e2x + 1.0);
      }
    }
  }

  void tanhBackward(String nameOutData, String nameGradOut, String nameGradIn) {
    DartCpuElement outData = memory[nameOutData]!;
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < outData.totalElements; i = i + 1) {
      double t = outData.data[i];
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * (1.0 - t * t);
    }
  }

  double _erf(double x) {
    double sign = x < 0.0 ? -1.0 : 1.0;
    double ax = x.abs();
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double poly = t *
        (0.254829592 +
            t *
                (-0.284496736 +
                    t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    double y = 1.0 - poly * exp(-ax * ax);
    return sign * y;
  }

  void geluForward(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double x = inElem.data[i];
      outElem.data[i] = 0.5 * x * (1.0 + _erf(x * 0.707106781));
    }
  }

  void geluBackward(String nameIn, String nameGradOut, String nameGradIn) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradIn = memory[nameGradIn]!;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      double x = inElem.data[i];
      double cdf = 0.5 * (1.0 + _erf(x * 0.707106781));
      double pdf = 0.3989422804 * exp(-0.5 * x * x);
      gradIn.data[i] = gradIn.data[i] + gradOut.data[i] * (cdf + x * pdf);
    }
  }

  void softmaxForward(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int rows = inElem.sizes.length == 1 ? 1 : inElem.sizes[0];
    int cols = inElem.sizes.length == 1 ? inElem.sizes[0] : inElem.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;

      double maxVal = inElem.data[rowOffset];
      for (int c = 1; c < cols; c = c + 1) {
        double val = inElem.data[rowOffset + c];
        if (val > maxVal) {
          maxVal = val;
        }
      }

      double sumExps = 0.0;
      for (int c = 0; c < cols; c = c + 1) {
        double e = exp(inElem.data[rowOffset + c] - maxVal);
        outElem.data[rowOffset + c] = e;
        sumExps = sumExps + e;
      }

      for (int c = 0; c < cols; c = c + 1) {
        outElem.data[rowOffset + c] = outElem.data[rowOffset + c] / sumExps;
      }
    }
  }

  void softmaxBackward(
      String nameGradOut, String nameOutData, String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement outData = memory[nameOutData]!;
    DartCpuElement gradIn = memory[nameGradIn]!;

    int rows = gradOut.sizes.length == 1 ? 1 : gradOut.sizes[0];
    int cols = gradOut.sizes.length == 1 ? gradOut.sizes[0] : gradOut.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;

      double dotProd = 0.0;
      for (int c = 0; c < cols; c = c + 1) {
        dotProd =
            dotProd + gradOut.data[rowOffset + c] * outData.data[rowOffset + c];
      }

      for (int c = 0; c < cols; c = c + 1) {
        int idx = rowOffset + c;
        gradIn.data[idx] = gradIn.data[idx] +
            outData.data[idx] * (gradOut.data[idx] - dotProd);
      }
    }
  }

  void mseLossForward(String predName, String targetName, String outName) {
    DartCpuElement pred = memory[predName]!;
    DartCpuElement target = memory[targetName]!;
    DartCpuElement out = memory[outName]!;

    int n = pred.totalElements;
    double sum = 0.0;
    for (int i = 0; i < n; i = i + 1) {
      double diff = pred.data[i] - target.data[i];
      sum = sum + diff * diff;
    }
    out.data[0] = sum / n;
  }

  void mseLossBackward(String predName, String targetName, String gradInName,
      String gradOutName) {
    DartCpuElement pred = memory[predName]!;
    DartCpuElement target = memory[targetName]!;
    DartCpuElement gradIn = memory[gradInName]!;
    DartCpuElement gradOut = memory[gradOutName]!;

    int n = pred.totalElements;
    double g = gradOut.data[0];
    for (int i = 0; i < n; i = i + 1) {
      gradIn.data[i] =
          gradIn.data[i] + g * 2.0 * (pred.data[i] - target.data[i]) / n;
    }
  }

  void bceLossForward(String namePred, String nameTarget, String nameOut) {
    DartCpuElement pred = memory[namePred]!;
    DartCpuElement target = memory[nameTarget]!;
    DartCpuElement out = memory[nameOut]!;

    int n = pred.totalElements;
    double sum = 0.0;
    for (int i = 0; i < n; i = i + 1) {
      double p = pred.data[i];
      if (p < 1e-9) {
        p = 1e-9;
      }
      if (p > 0.999999) {
        p = 0.999999;
      }
      double t = target.data[i];
      sum = sum - (t * log(p) + (1.0 - t) * log(1.0 - p));
    }
    out.data[0] = sum / n;
  }

  void bceLossBackward(String nameGradOut, String namePred, String nameTarget,
      String nameGradIn) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement pred = memory[namePred]!;
    DartCpuElement target = memory[nameTarget]!;
    DartCpuElement gradIn = memory[nameGradIn]!;

    int n = pred.totalElements;
    double g = gradOut.data[0];
    for (int i = 0; i < n; i = i + 1) {
      double p = pred.data[i];
      if (p < 1e-9) {
        p = 1e-9;
      }
      if (p > 0.999999) {
        p = 0.999999;
      }
      gradIn.data[i] =
          gradIn.data[i] + g * ((p - target.data[i]) / (p * (1.0 - p)));
    }
  }

  void sgdUpdate(String dataName, String gradName, double learningRate) {
    DartCpuElement dataElem = memory[dataName]!;
    DartCpuElement gradElem = memory[gradName]!;
    for (int i = 0; i < dataElem.totalElements; i = i + 1) {
      dataElem.data[i] = dataElem.data[i] - (learningRate * gradElem.data[i]);
    }
  }

  void adamUpdate(
      String dataName,
      String gradName,
      String mName,
      String vName,
      double learningRate,
      double beta1,
      double beta2,
      double eps,
      int step,
      double weightDecay) {
    DartCpuElement dataElem = memory[dataName]!;
    DartCpuElement gradElem = memory[gradName]!;
    DartCpuElement mElem = memory[mName]!;
    DartCpuElement vElem = memory[vName]!;

    double oneMinusB1t = 1.0 - pow(beta1, step).toDouble();
    double oneMinusB2t = 1.0 - pow(beta2, step).toDouble();

    for (int i = 0; i < dataElem.totalElements; i = i + 1) {
      double g = gradElem.data[i];
      double mi = beta1 * mElem.data[i] + (1.0 - beta1) * g;
      double vi = beta2 * vElem.data[i] + (1.0 - beta2) * (g * g);
      mElem.data[i] = mi;
      vElem.data[i] = vi;

      double mHat = mi / oneMinusB1t;
      double vHat = vi / oneMinusB2t;

      dataElem.data[i] =
          dataElem.data[i] - learningRate * (mHat / (sqrt(vHat) + eps));
      if (weightDecay != 0.0) {
        dataElem.data[i] =
            dataElem.data[i] - learningRate * weightDecay * dataElem.data[i];
      }
    }
  }

  void clipGradValue(String bufferName, double clipValue) {
    DartCpuElement elem = memory[bufferName]!;
    double absClipVal = clipValue.abs();
    for (int i = 0; i < elem.totalElements; i = i + 1) {
      double val = elem.data[i];
      if (val < -absClipVal) {
        val = -absClipVal;
      }
      if (val > absClipVal) {
        val = absClipVal;
      }
      elem.data[i] = val;
    }
  }

  void matmulBiasReluForward(String nameX, String nameW, String nameB,
      String nameReluOut, String namePreReluOut) {
    DartCpuElement xElem = memory[nameX]!;
    DartCpuElement wElem = memory[nameW]!;
    DartCpuElement bElem = memory[nameB]!;
    DartCpuElement reluOut = memory[nameReluOut]!;
    DartCpuElement preReluOut = memory[namePreReluOut]!;

    int m = xElem.sizes[0];
    int k = xElem.sizes[1];
    int n = wElem.sizes[1];

    for (int r = 0; r < m; r = r + 1) {
      for (int c = 0; c < n; c = c + 1) {
        double sum = 0.0;
        for (int p = 0; p < k; p = p + 1) {
          sum = sum + xElem.data[r * k + p] * wElem.data[p * n + c];
        }
        double val = sum + bElem.data[c];
        int outIdx = r * n + c;
        preReluOut.data[outIdx] = val;
        reluOut.data[outIdx] = val > 0.0 ? val : 0.0;
      }
    }
  }

  void layerNormForward(String nameIn, String nameGamma, String nameBeta,
      String nameOut, String nameMean, String nameRstd, double epsilon) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement betaElem = memory[nameBeta]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement meanElem = memory[nameMean]!;
    DartCpuElement rstdElem = memory[nameRstd]!;

    int rows = inElem.sizes[0];
    int cols = inElem.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;

      double sum = 0.0;
      for (int c = 0; c < cols; c = c + 1) {
        sum = sum + inElem.data[rowOffset + c];
      }
      double mean = sum / cols;

      double varSum = 0.0;
      for (int c = 0; c < cols; c = c + 1) {
        double diff = inElem.data[rowOffset + c] - mean;
        varSum = varSum + diff * diff;
      }
      double variance = varSum / cols;
      double rstd = 1.0 / sqrt(variance + epsilon);

      meanElem.data[r] = mean;
      rstdElem.data[r] = rstd;

      for (int c = 0; c < cols; c = c + 1) {
        int idx = rowOffset + c;
        double xHat = (inElem.data[idx] - mean) * rstd;
        outElem.data[idx] = xHat * gammaElem.data[c] + betaElem.data[c];
      }
    }
  }

  void layerNormBackward(
      String nameGradOut,
      String nameIn,
      String nameGamma,
      String nameMean,
      String nameRstd,
      String nameGradIn,
      String nameGradGamma,
      String nameGradBeta) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement meanElem = memory[nameMean]!;
    DartCpuElement rstdElem = memory[nameRstd]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;
    DartCpuElement gradGammaElem = memory[nameGradGamma]!;
    DartCpuElement gradBetaElem = memory[nameGradBeta]!;

    int rows = inElem.sizes[0];
    int cols = inElem.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;
      double mean = meanElem.data[r];
      double rstd = rstdElem.data[r];

      double sumGradXHat = 0.0;
      double sumGradXHatXHat = 0.0;

      for (int c = 0; c < cols; c = c + 1) {
        int idx = rowOffset + c;
        double xHat = (inElem.data[idx] - mean) * rstd;
        double gOut = gradOutElem.data[idx];
        double gradXHat = gOut * gammaElem.data[c];

        sumGradXHat = sumGradXHat + gradXHat;
        sumGradXHatXHat = sumGradXHatXHat + gradXHat * xHat;

        gradBetaElem.data[c] = gradBetaElem.data[c] + gOut;
        gradGammaElem.data[c] = gradGammaElem.data[c] + gOut * xHat;
      }

      for (int c = 0; c < cols; c = c + 1) {
        int idx = rowOffset + c;
        double xHat = (inElem.data[idx] - mean) * rstd;
        double gradXHat = gradOutElem.data[idx] * gammaElem.data[c];

        double term1 = cols * gradXHat;
        double term2 = sumGradXHat;
        double term3 = xHat * sumGradXHatXHat;

        gradInElem.data[idx] =
            gradInElem.data[idx] + (rstd / cols) * (term1 - term2 - term3);
      }
    }
  }

  void conv2dForward(String nameIn, String nameKernel, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement kernelElem = memory[nameKernel]!;
    DartCpuElement outElem = memory[nameOut]!;

    int inWidth = inElem.sizes[1];
    int kernelHeight = kernelElem.sizes[0];
    int kernelWidth = kernelElem.sizes[1];
    int outHeight = outElem.sizes[0];
    int outWidth = outElem.sizes[1];

    for (int y = 0; y < outHeight; y = y + 1) {
      for (int x = 0; x < outWidth; x = x + 1) {
        double sum = 0.0;
        for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
          for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
            int inIdx = (y + ky) * inWidth + (x + kx);
            int kIdx = ky * kernelWidth + kx;
            sum = sum + inElem.data[inIdx] * kernelElem.data[kIdx];
          }
        }
        outElem.data[y * outWidth + x] = sum;
      }
    }
  }

  void conv2dBackwardInput(
      String nameGradOut, String nameKernel, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement kernelElem = memory[nameKernel]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int inWidth = gradInElem.sizes[1];
    int kernelHeight = kernelElem.sizes[0];
    int kernelWidth = kernelElem.sizes[1];
    int outHeight = gradOutElem.sizes[0];
    int outWidth = gradOutElem.sizes[1];

    for (int y = 0; y < outHeight; y = y + 1) {
      for (int x = 0; x < outWidth; x = x + 1) {
        double grad = gradOutElem.data[y * outWidth + x];
        for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
          for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
            int inIdx = (y + ky) * inWidth + (x + kx);
            int kIdx = ky * kernelWidth + kx;
            gradInElem.data[inIdx] =
                gradInElem.data[inIdx] + kernelElem.data[kIdx] * grad;
          }
        }
      }
    }
  }

  void conv2dBackwardKernel(
      String nameIn, String nameGradOut, String nameGradKernel) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradKernelElem = memory[nameGradKernel]!;

    int inWidth = inElem.sizes[1];
    int kernelHeight = gradKernelElem.sizes[0];
    int kernelWidth = gradKernelElem.sizes[1];
    int outHeight = gradOutElem.sizes[0];
    int outWidth = gradOutElem.sizes[1];

    for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
      for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
        double sum = 0.0;
        for (int y = 0; y < outHeight; y = y + 1) {
          for (int x = 0; x < outWidth; x = x + 1) {
            int inIdx = (y + ky) * inWidth + (x + kx);
            int outIdx = y * outWidth + x;
            sum = sum + inElem.data[inIdx] * gradOutElem.data[outIdx];
          }
        }
        int kIdx = ky * kernelWidth + kx;
        gradKernelElem.data[kIdx] = gradKernelElem.data[kIdx] + sum;
      }
    }
  }

  void conv2dMultiForward(
      String nameIn,
      String nameWeight,
      String nameBias,
      String nameOut,
      int cIn,
      int cOut,
      int kH,
      int kW,
      int padT,
      int padL,
      int strideH,
      int strideW) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement weightElem = memory[nameWeight]!;
    DartCpuElement biasElem = memory[nameBias]!;
    DartCpuElement outElem = memory[nameOut]!;

    int inH = inElem.sizes.length == 3 ? inElem.sizes[1] : inElem.sizes[0];
    int inW = inElem.sizes.length == 3 ? inElem.sizes[2] : inElem.sizes[1];
    int outH = outElem.sizes[1];
    int outW = outElem.sizes[2];

    for (int outC = 0; outC < cOut; outC = outC + 1) {
      for (int outY = 0; outY < outH; outY = outY + 1) {
        for (int outX = 0; outX < outW; outX = outX + 1) {
          double sum = biasElem.data[outC];
          for (int inC = 0; inC < cIn; inC = inC + 1) {
            for (int ky = 0; ky < kH; ky = ky + 1) {
              int inY = (outY * strideH) + ky - padT;
              if (inY < 0 || inY >= inH) {
                continue;
              }
              for (int kx = 0; kx < kW; kx = kx + 1) {
                int inX = (outX * strideW) + kx - padL;
                if (inX < 0 || inX >= inW) {
                  continue;
                }
                int inIdx = inC * (inH * inW) + inY * inW + inX;
                int wIdx =
                    outC * (cIn * kH * kW) + inC * (kH * kW) + ky * kW + kx;
                sum = sum + inElem.data[inIdx] * weightElem.data[wIdx];
              }
            }
          }
          int outIdx = outC * (outH * outW) + outY * outW + outX;
          outElem.data[outIdx] = sum;
        }
      }
    }
  }

  void conv2dMultiBackwardInput(
      String nameGradOut,
      String nameWeight,
      String nameGradIn,
      int cIn,
      int cOut,
      int kH,
      int kW,
      int padT,
      int padL,
      int strideH,
      int strideW) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement weightElem = memory[nameWeight]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int inH = gradInElem.sizes.length == 3
        ? gradInElem.sizes[1]
        : gradInElem.sizes[0];
    int inW = gradInElem.sizes.length == 3
        ? gradInElem.sizes[2]
        : gradInElem.sizes[1];
    int outH = gradOutElem.sizes[1];
    int outW = gradOutElem.sizes[2];

    for (int inC = 0; inC < cIn; inC = inC + 1) {
      for (int inY = 0; inY < inH; inY = inY + 1) {
        for (int inX = 0; inX < inW; inX = inX + 1) {
          double sum = 0.0;
          for (int outC = 0; outC < cOut; outC = outC + 1) {
            for (int ky = 0; ky < kH; ky = ky + 1) {
              int outYRaw = inY - ky + padT;
              if (outYRaw < 0 || outYRaw % strideH != 0) {
                continue;
              }
              int outY = outYRaw ~/ strideH;
              if (outY >= outH) {
                continue;
              }
              for (int kx = 0; kx < kW; kx = kx + 1) {
                int outXRaw = inX - kx + padL;
                if (outXRaw < 0 || outXRaw % strideW != 0) {
                  continue;
                }
                int outX = outXRaw ~/ strideW;
                if (outX >= outW) {
                  continue;
                }
                int gradOutIdx = outC * (outH * outW) + outY * outW + outX;
                int wIdx =
                    outC * (cIn * kH * kW) + inC * (kH * kW) + ky * kW + kx;
                sum =
                    sum + gradOutElem.data[gradOutIdx] * weightElem.data[wIdx];
              }
            }
          }
          int inIdx = inC * (inH * inW) + inY * inW + inX;
          gradInElem.data[inIdx] = gradInElem.data[inIdx] + sum;
        }
      }
    }
  }

  void conv2dMultiBackwardWeight(
      String nameInput,
      String nameGradOut,
      String nameGradWeight,
      String nameGradBias,
      int cIn,
      int cOut,
      int kH,
      int kW,
      int padT,
      int padL,
      int strideH,
      int strideW) {
    DartCpuElement inElem = memory[nameInput]!;
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradWeightElem = memory[nameGradWeight]!;
    DartCpuElement gradBiasElem = memory[nameGradBias]!;

    int inH = inElem.sizes.length == 3 ? inElem.sizes[1] : inElem.sizes[0];
    int inW = inElem.sizes.length == 3 ? inElem.sizes[2] : inElem.sizes[1];
    int outH = gradOutElem.sizes[1];
    int outW = gradOutElem.sizes[2];

    for (int outC = 0; outC < cOut; outC = outC + 1) {
      for (int inC = 0; inC < cIn; inC = inC + 1) {
        for (int ky = 0; ky < kH; ky = ky + 1) {
          for (int kx = 0; kx < kW; kx = kx + 1) {
            double sum = 0.0;
            for (int outY = 0; outY < outH; outY = outY + 1) {
              int inY = (outY * strideH) + ky - padT;
              if (inY < 0 || inY >= inH) {
                continue;
              }
              for (int outX = 0; outX < outW; outX = outX + 1) {
                int inX = (outX * strideW) + kx - padL;
                if (inX < 0 || inX >= inW) {
                  continue;
                }
                int inIdx = inC * (inH * inW) + inY * inW + inX;
                int gradOutIdx = outC * (outH * outW) + outY * outW + outX;
                sum = sum + inElem.data[inIdx] * gradOutElem.data[gradOutIdx];
              }
            }
            int wIdx = outC * (cIn * kH * kW) + inC * (kH * kW) + ky * kW + kx;
            gradWeightElem.data[wIdx] = gradWeightElem.data[wIdx] + sum;
          }
        }
      }
    }

    int area = outH * outW;
    for (int outC = 0; outC < cOut; outC = outC + 1) {
      double sum = 0.0;
      int offset = outC * area;
      for (int i = 0; i < area; i = i + 1) {
        sum = sum + gradOutElem.data[offset + i];
      }
      gradBiasElem.data[outC] = gradBiasElem.data[outC] + sum;
    }
  }

  void im2col(
      String nameIn, String nameOutCol, int kernelHeight, int kernelWidth) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outColElem = memory[nameOutCol]!;

    int inHeight = inElem.sizes[0];
    int inWidth = inElem.sizes[1];
    int outHeight = inHeight - kernelHeight + 1;
    int outWidth = inWidth - kernelWidth + 1;
    int outArea = outHeight * outWidth;

    for (int y = 0; y < outHeight; y = y + 1) {
      for (int x = 0; x < outWidth; x = x + 1) {
        int colIdx = y * outWidth + x;
        for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
          for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
            int rowIdx = ky * kernelWidth + kx;
            int outIdx = rowIdx * outArea + colIdx;
            int inIdx = (y + ky) * inWidth + (x + kx);
            outColElem.data[outIdx] = inElem.data[inIdx];
          }
        }
      }
    }
  }

  void col2im(String nameColGrad, String nameInGrad, int kernelHeight,
      int kernelWidth) {
    DartCpuElement colGradElem = memory[nameColGrad]!;
    DartCpuElement inGradElem = memory[nameInGrad]!;

    int inHeight = inGradElem.sizes[0];
    int inWidth = inGradElem.sizes[1];
    int outHeight = inHeight - kernelHeight + 1;
    int outWidth = inWidth - kernelWidth + 1;
    int outArea = outHeight * outWidth;

    for (int y = 0; y < outHeight; y = y + 1) {
      for (int x = 0; x < outWidth; x = x + 1) {
        int colIdx = y * outWidth + x;
        for (int ky = 0; ky < kernelHeight; ky = ky + 1) {
          for (int kx = 0; kx < kernelWidth; kx = kx + 1) {
            int rowIdx = ky * kernelWidth + kx;
            int colBufIdx = rowIdx * outArea + colIdx;
            int inIdx = (y + ky) * inWidth + (x + kx);
            inGradElem.data[inIdx] =
                inGradElem.data[inIdx] + colGradElem.data[colBufIdx];
          }
        }
      }
    }
  }

  void globalAvgPoolForward(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int seqLen = inElem.sizes[0];
    int dModel = inElem.sizes[1];

    for (int col = 0; col < dModel; col = col + 1) {
      double sum = 0.0;
      for (int r = 0; r < seqLen; r = r + 1) {
        sum = sum + inElem.data[r * dModel + col];
      }
      outElem.data[col] = sum / seqLen;
    }
  }

  void globalAvgPoolBackward(String nameGradOut, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int seqLen = gradInElem.sizes[0];
    int dModel = gradInElem.sizes[1];

    for (int r = 0; r < seqLen; r = r + 1) {
      for (int col = 0; col < dModel; col = col + 1) {
        int idx = r * dModel + col;
        gradInElem.data[idx] =
            gradInElem.data[idx] + gradOutElem.data[col] / seqLen;
      }
    }
  }

  void maxPool1dForward(String nameIn, String nameOut, String nameIndices,
      int poolSize, int stride) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement indicesElem = memory[nameIndices]!;

    int inLen = inElem.sizes[0];
    int outLen = outElem.sizes[0];

    for (int outIdx = 0; outIdx < outLen; outIdx = outIdx + 1) {
      int startIdx = outIdx * stride;
      double maxVal = -double.infinity;
      int maxIdx = -1;

      for (int p = 0; p < poolSize; p = p + 1) {
        int inIdx = startIdx + p;
        if (inIdx < inLen) {
          double val = inElem.data[inIdx];
          if (val > maxVal) {
            maxVal = val;
            maxIdx = inIdx;
          }
        }
      }

      outElem.data[outIdx] = maxVal;
      indicesElem.data[outIdx] = maxIdx.toDouble();
    }
  }

  void maxPool1dBackward(
      String nameGradOut, String nameIndices, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement indicesElem = memory[nameIndices]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int outLen = gradOutElem.sizes[0];

    for (int outIdx = 0; outIdx < outLen; outIdx = outIdx + 1) {
      int maxIdx = indicesElem.data[outIdx].toInt();
      if (maxIdx >= 0) {
        gradInElem.data[maxIdx] =
            gradInElem.data[maxIdx] + gradOutElem.data[outIdx];
      }
    }
  }

  void maxPool2dForward(String nameIn, String nameOut, String nameIndices,
      int poolSize, int stride) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement indicesElem = memory[nameIndices]!;

    int inH = inElem.sizes[0];
    int inW = inElem.sizes[1];
    int outH = outElem.sizes[0];
    int outW = outElem.sizes[1];

    for (int outY = 0; outY < outH; outY = outY + 1) {
      for (int outX = 0; outX < outW; outX = outX + 1) {
        int outIdx = outY * outW + outX;
        int startY = outY * stride;
        int startX = outX * stride;

        double maxVal = -double.infinity;
        int maxIdx = -1;

        for (int py = 0; py < poolSize; py = py + 1) {
          for (int px = 0; px < poolSize; px = px + 1) {
            int inY = startY + py;
            int inX = startX + px;
            if (inY < inH && inX < inW) {
              int inIdx = inY * inW + inX;
              double val = inElem.data[inIdx];
              if (val > maxVal) {
                maxVal = val;
                maxIdx = inIdx;
              }
            }
          }
        }

        outElem.data[outIdx] = maxVal;
        indicesElem.data[outIdx] = maxIdx.toDouble();
      }
    }
  }

  void maxPool2dBackward(
      String nameGradOut, String nameIndices, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement indicesElem = memory[nameIndices]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int outTotal = gradOutElem.totalElements;

    for (int outIdx = 0; outIdx < outTotal; outIdx = outIdx + 1) {
      int maxIdx = indicesElem.data[outIdx].toInt();
      if (maxIdx >= 0) {
        gradInElem.data[maxIdx] =
            gradInElem.data[maxIdx] + gradOutElem.data[outIdx];
      }
    }
  }

  void avgPool2dForward(
      String nameIn, String nameOut, int poolSize, int stride) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int inH = inElem.sizes[0];
    int inW = inElem.sizes[1];
    int outH = outElem.sizes[0];
    int outW = outElem.sizes[1];
    double numElements = (poolSize * poolSize).toDouble();

    for (int outY = 0; outY < outH; outY = outY + 1) {
      for (int outX = 0; outX < outW; outX = outX + 1) {
        int outIdx = outY * outW + outX;
        int startY = outY * stride;
        int startX = outX * stride;

        double sum = 0.0;
        for (int py = 0; py < poolSize; py = py + 1) {
          for (int px = 0; px < poolSize; px = px + 1) {
            int inY = startY + py;
            int inX = startX + px;
            if (inY < inH && inX < inW) {
              sum = sum + inElem.data[inY * inW + inX];
            }
          }
        }
        outElem.data[outIdx] = sum / numElements;
      }
    }
  }

  void avgPool2dBackward(
      String nameGradOut, String nameGradIn, int poolSize, int stride) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int inH = gradInElem.sizes[0];
    int inW = gradInElem.sizes[1];
    int outH = gradOutElem.sizes[0];
    int outW = gradOutElem.sizes[1];
    double numElements = (poolSize * poolSize).toDouble();

    for (int outY = 0; outY < outH; outY = outY + 1) {
      for (int outX = 0; outX < outW; outX = outX + 1) {
        int outIdx = outY * outW + outX;
        double g = gradOutElem.data[outIdx] / numElements;
        int startY = outY * stride;
        int startX = outX * stride;

        for (int py = 0; py < poolSize; py = py + 1) {
          for (int px = 0; px < poolSize; px = px + 1) {
            int inY = startY + py;
            int inX = startX + px;
            if (inY < inH && inX < inW) {
              int inIdx = inY * inW + inX;
              gradInElem.data[inIdx] = gradInElem.data[inIdx] + g;
            }
          }
        }
      }
    }
  }

  void batchNorm1dForward(
      String nameIn,
      String nameGamma,
      String nameBeta,
      String nameRunningMean,
      String nameRunningVar,
      String nameOut,
      String nameSavedMean,
      String nameSavedInvVar,
      double momentum,
      double epsilon,
      bool isTraining) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement betaElem = memory[nameBeta]!;
    DartCpuElement runningMeanElem = memory[nameRunningMean]!;
    DartCpuElement runningVarElem = memory[nameRunningVar]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement savedMeanElem = memory[nameSavedMean]!;
    DartCpuElement savedInvVarElem = memory[nameSavedInvVar]!;

    int numFeatures = inElem.sizes[0];

    for (int i = 0; i < numFeatures; i = i + 1) {
      if (isTraining) {
        runningMeanElem.data[i] = momentum * runningMeanElem.data[i] +
            (1.0 - momentum) * inElem.data[i];
        runningVarElem.data[i] = momentum * runningVarElem.data[i];
      }

      double meanToUse = runningMeanElem.data[i];
      double varToUse = runningVarElem.data[i];
      double invStd = 1.0 / sqrt(varToUse + epsilon);

      savedMeanElem.data[i] = meanToUse;
      savedInvVarElem.data[i] = invStd;

      double xHat = (inElem.data[i] - meanToUse) * invStd;
      outElem.data[i] = gammaElem.data[i] * xHat + betaElem.data[i];
    }
  }

  void batchNorm1dBackward(
      String nameGradOut,
      String nameIn,
      String nameGamma,
      String nameSavedMean,
      String nameSavedInvVar,
      String nameGradIn,
      String nameGradGamma,
      String nameGradBeta) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement savedMeanElem = memory[nameSavedMean]!;
    DartCpuElement savedInvVarElem = memory[nameSavedInvVar]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;
    DartCpuElement gradGammaElem = memory[nameGradGamma]!;
    DartCpuElement gradBetaElem = memory[nameGradBeta]!;

    int numFeatures = inElem.sizes[0];

    for (int i = 0; i < numFeatures; i = i + 1) {
      double invStd = savedInvVarElem.data[i];
      double xHat = (inElem.data[i] - savedMeanElem.data[i]) * invStd;
      double gOut = gradOutElem.data[i];

      gradGammaElem.data[i] = gradGammaElem.data[i] + gOut * xHat;
      gradBetaElem.data[i] = gradBetaElem.data[i] + gOut;
      gradInElem.data[i] =
          gradInElem.data[i] + gOut * gammaElem.data[i] * invStd;
    }
  }

  void batchNorm2dForward(
      String nameIn,
      String nameGamma,
      String nameBeta,
      String nameRunningMean,
      String nameRunningVar,
      String nameOut,
      String nameSavedMean,
      String nameSavedInvVar,
      double momentum,
      double epsilon,
      bool isTraining) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement betaElem = memory[nameBeta]!;
    DartCpuElement runningMeanElem = memory[nameRunningMean]!;
    DartCpuElement runningVarElem = memory[nameRunningVar]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement savedMeanElem = memory[nameSavedMean]!;
    DartCpuElement savedInvVarElem = memory[nameSavedInvVar]!;

    int numChannels = inElem.sizes[0];
    int height = inElem.sizes[1];
    int width = inElem.sizes[2];
    int elementsPerChannel = height * width;

    for (int c = 0; c < numChannels; c = c + 1) {
      int cOffset = c * elementsPerChannel;

      double sum = 0.0;
      for (int i = 0; i < elementsPerChannel; i = i + 1) {
        sum = sum + inElem.data[cOffset + i];
      }
      double currentMean = sum / elementsPerChannel;

      double varSum = 0.0;
      for (int i = 0; i < elementsPerChannel; i = i + 1) {
        double diff = inElem.data[cOffset + i] - currentMean;
        varSum = varSum + diff * diff;
      }
      double currentVar = varSum / elementsPerChannel;

      if (isTraining) {
        runningMeanElem.data[c] =
            momentum * runningMeanElem.data[c] + (1.0 - momentum) * currentMean;
        runningVarElem.data[c] =
            momentum * runningVarElem.data[c] + (1.0 - momentum) * currentVar;
      }

      double meanToUse = isTraining ? currentMean : runningMeanElem.data[c];
      double varToUse = isTraining ? currentVar : runningVarElem.data[c];
      double invStd = 1.0 / sqrt(varToUse + epsilon);

      savedMeanElem.data[c] = meanToUse;
      savedInvVarElem.data[c] = invStd;

      double g = gammaElem.data[c];
      double b = betaElem.data[c];

      for (int i = 0; i < elementsPerChannel; i = i + 1) {
        double xHat = (inElem.data[cOffset + i] - meanToUse) * invStd;
        outElem.data[cOffset + i] = g * xHat + b;
      }
    }
  }

  void batchNorm2dBackward(
      String nameGradOut,
      String nameIn,
      String nameGamma,
      String nameSavedMean,
      String nameSavedInvVar,
      String nameGradIn,
      String nameGradGamma,
      String nameGradBeta) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement gammaElem = memory[nameGamma]!;
    DartCpuElement savedMeanElem = memory[nameSavedMean]!;
    DartCpuElement savedInvVarElem = memory[nameSavedInvVar]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;
    DartCpuElement gradGammaElem = memory[nameGradGamma]!;
    DartCpuElement gradBetaElem = memory[nameGradBeta]!;

    int numChannels = inElem.sizes[0];
    int height = inElem.sizes[1];
    int width = inElem.sizes[2];
    int elementsPerChannel = height * width;

    for (int c = 0; c < numChannels; c = c + 1) {
      int cOffset = c * elementsPerChannel;
      double mean = savedMeanElem.data[c];
      double invStd = savedInvVarElem.data[c];
      double g = gammaElem.data[c];

      double sumGradGamma = 0.0;
      double sumGradBeta = 0.0;

      for (int i = 0; i < elementsPerChannel; i = i + 1) {
        int idx = cOffset + i;
        double gOut = gradOutElem.data[idx];
        double xHat = (inElem.data[idx] - mean) * invStd;

        sumGradGamma = sumGradGamma + gOut * xHat;
        sumGradBeta = sumGradBeta + gOut;

        gradInElem.data[idx] = gradInElem.data[idx] + gOut * g * invStd;
      }

      gradGammaElem.data[c] = gradGammaElem.data[c] + sumGradGamma;
      gradBetaElem.data[c] = gradBetaElem.data[c] + sumGradBeta;
    }
  }

  void dropoutForward(String nameIn, String nameOut, String nameMask,
      double dropRate, int seed) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement maskElem = memory[nameMask]!;

    double scale = 1.0 / (1.0 - dropRate);

    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      int state = (seed + i * 0x853c49e6) & 0xFFFFFFFF;
      state = (state ^ (state >> 13)) & 0xFFFFFFFF;
      state = (state * 0x0815451) & 0xFFFFFFFF;
      state = (state ^ (state >> 11)) & 0xFFFFFFFF;
      double randVal = (state & 0x00ffffff) / 0x01000000;

      if (randVal < dropRate) {
        outElem.data[i] = 0.0;
        maskElem.data[i] = 0.0;
      } else {
        outElem.data[i] = inElem.data[i] * scale;
        maskElem.data[i] = scale;
      }
    }
  }

  void dropoutBackward(String nameGradOut, String nameMask, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement maskElem = memory[nameMask]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    for (int i = 0; i < gradOutElem.totalElements; i = i + 1) {
      gradInElem.data[i] =
          gradInElem.data[i] + gradOutElem.data[i] * maskElem.data[i];
    }
  }

  void markovCount(
      String nameSequence, String nameCountTable, int order, int numStates) {
    DartCpuElement seqElem = memory[nameSequence]!;
    DartCpuElement countElem = memory[nameCountTable]!;

    int seqLen = seqElem.totalElements;

    for (int i = 0; i < seqLen - order; i = i + 1) {
      int historyIdx = 0;
      for (int k = 0; k < order; k = k + 1) {
        historyIdx = (historyIdx * 2) + seqElem.data[i + k].toInt();
      }
      int nextState = seqElem.data[i + order].toInt();
      int tableIdx = historyIdx * numStates + nextState;
      countElem.data[tableIdx] = countElem.data[tableIdx] + 1.0;
    }
  }

  void markovNormalize(String nameCountTable, String nameProbTable,
      int numHistories, int numStates) {
    DartCpuElement countElem = memory[nameCountTable]!;
    DartCpuElement probElem = memory[nameProbTable]!;

    for (int histIdx = 0; histIdx < numHistories; histIdx = histIdx + 1) {
      int rowOffset = histIdx * numStates;
      double sum = 0.0;
      for (int s = 0; s < numStates; s = s + 1) {
        sum = sum + countElem.data[rowOffset + s];
      }
      for (int s = 0; s < numStates; s = s + 1) {
        int idx = rowOffset + s;
        if (sum > 0.0) {
          probElem.data[idx] = countElem.data[idx] / sum;
        } else {
          probElem.data[idx] = 1.0 / numStates;
        }
      }
    }
  }

  void markovPredict(String nameHistory, String nameProbTable,
      String nameOutProbs, int order, int numStates) {
    DartCpuElement histElem = memory[nameHistory]!;
    DartCpuElement probElem = memory[nameProbTable]!;
    DartCpuElement outElem = memory[nameOutProbs]!;

    int batchSize = histElem.sizes[0];

    for (int b = 0; b < batchSize; b = b + 1) {
      int histIdx = 0;
      int histOffset = b * order;
      for (int k = 0; k < order; k = k + 1) {
        histIdx = (histIdx * 2) + histElem.data[histOffset + k].toInt();
      }
      int outOffset = b * numStates;
      int probOffset = histIdx * numStates;
      for (int s = 0; s < numStates; s = s + 1) {
        outElem.data[outOffset + s] = probElem.data[probOffset + s];
      }
    }
  }

  void sliceRow(String nameIn, String nameOut, int rowIndex) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int sliceSize = outElem.totalElements;
    int offset = rowIndex * sliceSize;
    for (int i = 0; i < sliceSize; i = i + 1) {
      outElem.data[i] = inElem.data[offset + i];
    }
  }

  void sliceRowBackward(String nameGradOut, String nameGradIn, int rowIndex) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int sliceSize = gradOutElem.totalElements;
    int offset = rowIndex * sliceSize;
    for (int i = 0; i < sliceSize; i = i + 1) {
      gradInElem.data[offset + i] =
          gradInElem.data[offset + i] + gradOutElem.data[i];
    }
  }

  void sliceColumn(String nameIn, String nameOut, int startCol, int endCol) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int inRows = inElem.sizes[0];
    int inCols = inElem.sizes[1];
    int outCols = outElem.sizes[1];

    for (int r = 0; r < inRows; r = r + 1) {
      for (int c = 0; c < outCols; c = c + 1) {
        outElem.data[r * outCols + c] =
            inElem.data[r * inCols + (c + startCol)];
      }
    }
  }

  void sliceColumnBackward(
      String nameGradOut, String nameGradIn, int startCol, int endCol) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    int inRows = gradInElem.sizes[0];
    int inCols = gradInElem.sizes[1];
    int outCols = gradOutElem.sizes[1];

    for (int r = 0; r < inRows; r = r + 1) {
      for (int c = 0; c < outCols; c = c + 1) {
        int inIdx = r * inCols + (c + startCol);
        gradInElem.data[inIdx] =
            gradInElem.data[inIdx] + gradOutElem.data[r * outCols + c];
      }
    }
  }

  void stackRows(List<String> namesIn, String nameOut, int axis) {
    DartCpuElement outElem = memory[nameOut]!;

    int offset = 0;
    for (int k = 0; k < namesIn.length; k = k + 1) {
      DartCpuElement inElem = memory[namesIn[k]]!;
      for (int i = 0; i < inElem.totalElements; i = i + 1) {
        outElem.data[offset + i] = inElem.data[i];
      }
      offset = offset + inElem.totalElements;
    }
  }

  void stackRowsBackward(
      String nameGradOut, List<String> namesGradIn, int axis) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;

    int offset = 0;
    for (int k = 0; k < namesGradIn.length; k = k + 1) {
      DartCpuElement gradInElem = memory[namesGradIn[k]]!;
      for (int i = 0; i < gradInElem.totalElements; i = i + 1) {
        gradInElem.data[i] = gradOutElem.data[offset + i];
      }
      offset = offset + gradInElem.totalElements;
    }
  }

  void concatenate(String nameA, String nameB, String nameOut, int axis) {
    DartCpuElement a = memory[nameA]!;
    DartCpuElement b = memory[nameB]!;
    DartCpuElement c = memory[nameOut]!;

    if (axis == 0) {
      int nA = a.totalElements;
      int nB = b.totalElements;
      for (int i = 0; i < nA; i = i + 1) {
        c.data[i] = a.data[i];
      }
      for (int i = 0; i < nB; i = i + 1) {
        c.data[nA + i] = b.data[i];
      }
      return;
    }

    int rows = a.sizes[0];
    int colsA = a.sizes[1];
    int colsB = b.sizes[1];
    int colsC = c.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      for (int col = 0; col < colsC; col = col + 1) {
        int outIdx = r * colsC + col;
        if (col < colsA) {
          c.data[outIdx] = a.data[r * colsA + col];
        } else {
          c.data[outIdx] = b.data[r * colsB + (col - colsA)];
        }
      }
    }
  }

  void concatenateBackward(String nameGradOut, String nameGradInA,
      String nameGradInB, int axis, int splitIndex) {
    DartCpuElement gradOut = memory[nameGradOut]!;
    DartCpuElement gradInA = memory[nameGradInA]!;
    DartCpuElement gradInB = memory[nameGradInB]!;

    if (axis == 0) {
      int nA = gradInA.totalElements;
      int nB = gradInB.totalElements;
      for (int i = 0; i < nA; i = i + 1) {
        gradInA.data[i] = gradInA.data[i] + gradOut.data[i];
      }
      for (int i = 0; i < nB; i = i + 1) {
        gradInB.data[i] = gradInB.data[i] + gradOut.data[nA + i];
      }
      return;
    }

    int rows = gradOut.sizes[0];
    int colsA = gradInA.sizes[1];
    int colsB = gradInB.sizes[1];
    int colsC = gradOut.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      for (int col = 0; col < colsC; col = col + 1) {
        double gradVal = gradOut.data[r * colsC + col];
        if (col < colsA) {
          int idxA = r * colsA + col;
          gradInA.data[idxA] = gradInA.data[idxA] + gradVal;
        } else {
          int idxB = r * colsB + (col - colsA);
          gradInB.data[idxB] = gradInB.data[idxB] + gradVal;
        }
      }
    }
  }

  void sumReduce(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    double sum = 0.0;
    for (int i = 0; i < inElem.totalElements; i = i + 1) {
      sum = sum + inElem.data[i];
    }
    outElem.data[0] = sum;
  }

  void sumReduceBackward(String nameGradOut, String nameGradIn) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement gradInElem = memory[nameGradIn]!;

    double g = gradOutElem.data[0];
    for (int i = 0; i < gradInElem.totalElements; i = i + 1) {
      gradInElem.data[i] = gradInElem.data[i] + g;
    }
  }

  void sumReduceColumns(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int rows = inElem.sizes[0];
    int cols = inElem.sizes[1];

    for (int c = 0; c < cols; c = c + 1) {
      outElem.data[c] = 0.0;
    }

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;
      for (int c = 0; c < cols; c = c + 1) {
        outElem.data[c] = outElem.data[c] + inElem.data[rowOffset + c];
      }
    }
  }

  void sumReduceRows(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int rows = inElem.sizes[0];
    int cols = inElem.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;
      for (int c = 0; c < cols; c = c + 1) {
        outElem.data[r] = outElem.data[r] + inElem.data[rowOffset + c];
      }
    }
  }

  void embeddingForward(String nameIndices, String nameWeight, String nameOut) {
    DartCpuElement indicesElem = memory[nameIndices]!;
    DartCpuElement weightElem = memory[nameWeight]!;
    DartCpuElement outElem = memory[nameOut]!;

    int numIndices = indicesElem.totalElements;
    int vocabSize = weightElem.sizes[0];
    int embeddingDim = weightElem.sizes[1];

    for (int i = 0; i < numIndices; i = i + 1) {
      int wordIdx = indicesElem.data[i].toInt();
      int outOffset = i * embeddingDim;
      if (wordIdx >= 0 && wordIdx < vocabSize) {
        int weightOffset = wordIdx * embeddingDim;
        for (int d = 0; d < embeddingDim; d = d + 1) {
          outElem.data[outOffset + d] = weightElem.data[weightOffset + d];
        }
      } else {
        for (int d = 0; d < embeddingDim; d = d + 1) {
          outElem.data[outOffset + d] = 0.0;
        }
      }
    }
  }

  void embeddingBackward(
      String nameGradOut, String nameIndices, String nameGradWeight) {
    DartCpuElement gradOutElem = memory[nameGradOut]!;
    DartCpuElement indicesElem = memory[nameIndices]!;
    DartCpuElement gradWeightElem = memory[nameGradWeight]!;

    int numIndices = indicesElem.totalElements;
    int vocabSize = gradWeightElem.sizes[0];
    int embeddingDim = gradWeightElem.sizes[1];

    for (int i = 0; i < numIndices; i = i + 1) {
      int wordIdx = indicesElem.data[i].toInt();
      if (wordIdx >= 0 && wordIdx < vocabSize) {
        int outOffset = i * embeddingDim;
        int weightOffset = wordIdx * embeddingDim;
        for (int d = 0; d < embeddingDim; d = d + 1) {
          gradWeightElem.data[weightOffset + d] =
              gradWeightElem.data[weightOffset + d] +
                  gradOutElem.data[outOffset + d];
        }
      }
    }
  }

  void rmsNormForward(String nameIn, String nameW, String nameOut,
      String nameRstd, double eps, bool scalePlusOne) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;
    DartCpuElement rstdElem = memory[nameRstd]!;
    bool hasWeight = nameW.isNotEmpty;
    DartCpuElement? wElem = hasWeight ? memory[nameW] : null;

    int rows = inElem.sizes[0];
    int cols = inElem.sizes[1];

    for (int r = 0; r < rows; r = r + 1) {
      int rowOffset = r * cols;

      double sumSq = 0.0;
      for (int c = 0; c < cols; c = c + 1) {
        double val = inElem.data[rowOffset + c];
        sumSq = sumSq + val * val;
      }

      double rstd = 1.0 / sqrt((sumSq / cols) + eps);
      rstdElem.data[r] = rstd;

      for (int c = 0; c < cols; c = c + 1) {
        int idx = rowOffset + c;
        double normVal = inElem.data[idx] * rstd;
        if (hasWeight) {
          double scale = scalePlusOne ? (1.0 + wElem!.data[c]) : wElem!.data[c];
          outElem.data[idx] = normVal * scale;
        } else {
          outElem.data[idx] = normVal;
        }
      }
    }
  }

  void causalMaskForward(
      String nameIn, String nameOut, int batch, int heads, int seq) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int totalMatrices = batch * heads;
    for (int m = 0; m < totalMatrices; m = m + 1) {
      int matOffset = m * seq * seq;
      for (int r = 0; r < seq; r = r + 1) {
        int rowOffset = matOffset + r * seq;
        for (int c = 0; c < seq; c = c + 1) {
          int idx = rowOffset + c;
          if (c > r) {
            outElem.data[idx] = -10000.0;
          } else {
            outElem.data[idx] = inElem.data[idx];
          }
        }
      }
    }
  }

  void ropeForward(String nameIn, String nameCos, String nameSin,
      String nameOut, int batch, int seq, int heads, int dim) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement cosElem = memory[nameCos]!;
    DartCpuElement sinElem = memory[nameSin]!;
    DartCpuElement outElem = memory[nameOut]!;

    int halfDim = dim ~/ 2;
    int cosCols = cosElem.sizes[1];

    for (int b = 0; b < batch; b = b + 1) {
      for (int s = 0; s < seq; s = s + 1) {
        int csOffset = s * cosCols;
        for (int h = 0; h < heads; h = h + 1) {
          int baseIdx = ((b * seq + s) * heads + h) * dim;
          for (int i = 0; i < halfDim; i = i + 1) {
            double x1 = inElem.data[baseIdx + i];
            double x2 = inElem.data[baseIdx + i + halfDim];
            double c = cosElem.data[csOffset + i];
            double sn = sinElem.data[csOffset + i];

            outElem.data[baseIdx + i] = x1 * c - x2 * sn;
            outElem.data[baseIdx + i + halfDim] = x2 * c + x1 * sn;
          }
        }
      }
    }
  }

  void crossEntropyForward(
      String nameLogits, String nameTargets, String nameOut) {
    DartCpuElement logitsElem = memory[nameLogits]!;
    DartCpuElement targetsElem = memory[nameTargets]!;
    DartCpuElement outElem = memory[nameOut]!;

    int batchSize = logitsElem.sizes[0];
    int vocabSize = logitsElem.sizes[1];

    for (int b = 0; b < batchSize; b = b + 1) {
      int rowOffset = b * vocabSize;

      double maxVal = logitsElem.data[rowOffset];
      for (int v = 1; v < vocabSize; v = v + 1) {
        double val = logitsElem.data[rowOffset + v];
        if (val > maxVal) {
          maxVal = val;
        }
      }

      double sumExp = 0.0;
      for (int v = 0; v < vocabSize; v = v + 1) {
        sumExp = sumExp + exp(logitsElem.data[rowOffset + v] - maxVal);
      }

      int targetIdx = targetsElem.data[b].toInt();
      double logSumExp = maxVal + log(sumExp);
      outElem.data[b] = logSumExp - logitsElem.data[rowOffset + targetIdx];
    }
  }

  void argmaxForward(String nameIn, String nameOut) {
    DartCpuElement inElem = memory[nameIn]!;
    DartCpuElement outElem = memory[nameOut]!;

    int batchSize = inElem.sizes[0];
    int vocabSize = inElem.sizes[1];

    for (int b = 0; b < batchSize; b = b + 1) {
      int rowOffset = b * vocabSize;
      int maxIdx = 0;
      double maxVal = inElem.data[rowOffset];

      for (int v = 1; v < vocabSize; v = v + 1) {
        double val = inElem.data[rowOffset + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }

      outElem.data[b] = maxIdx.toDouble();
    }
  }
}
