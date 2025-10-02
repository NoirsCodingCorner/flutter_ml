import 'dart:math';

import '../activationFunctions/gelu.dart';
import '../diagnosysTools/logger.dart';


class Node {
  final List<Tensor> inputs;
  final Function backwardFn;
  final String opName;
  final int cost; // NEW: Stores the computational cost

  Node(this.inputs, this.backwardFn, {this.opName = 'op', this.cost = 0});
}

class Tensor<T> {
  T value;

  late T grad;
  Node? creator; // A reference to the node that created this tensor

  Tensor(this.value, {Node? creator2}) {
    creator = creator2;
    if (value is Scalar) {
      grad = 0.0 as T;
    } else if (value is Vector) {
      Vector valAsList = value as Vector;
      grad = List<double>.filled(valAsList.length, 0.0) as T;
    } else if (value is Matrix) {
      Matrix valAsMatrix = value as Matrix;
      int numRows = valAsMatrix.length;
      int numCols = valAsMatrix.isNotEmpty ? valAsMatrix[0].length : 0;
      grad =
          List.generate(numRows, (_) => List<double>.filled(numCols, 0.0)) as T;
    }
  }

  void backward() {
    if (this.creator == null) {
      this.grad = 1.0 as T;
      return;
    }

    // A more robust topological sort using a stack
    List<Node> topo = [];
    Set<Node> visited = {};
    List<Node> stack = [];

    void buildTopo(Node? node) {
      if (node == null || visited.contains(node)) {
        return;
      }

      visited.add(node);
      for (var inputTensor in node.inputs) {
        if (inputTensor.creator != null) {
          buildTopo(inputTensor.creator!);
        }
      }
      topo.add(node);
    }

    buildTopo(this.creator!);

    this.grad = 1.0 as T;

    for (Node node in topo.reversed) {
      node.backwardFn();
    }
  }

  // Add this method to the Tensor class
  void zeroGrad() {
    if (grad is double) {
      grad = 0.0 as T;
    } else if (grad is List<double>) {
      for (int i = 0; i < (grad as List<double>).length; i++) {
        (grad as List<double>)[i] = 0.0;
      }
    } else if (grad is List<List<double>>) {
      for (int i = 0; i < (grad as List<List<double>>).length; i++) {
        for (int j = 0; j < (grad as List<List<double>>)[i].length; j++) {
          (grad as List<List<double>>)[i][j] = 0.0;
        }
      }
    }
  }


  /// Prints a color-coded, visual representation of the computational graph.
  ///
  /// - 🟡 **Yellow:** The final output tensor (the root of the graph).
  /// - 🟢 **Green:** Leaf nodes (initial inputs, weights, and biases).
  /// - 🔵 **Blue:** Intermediate tensors created by operations.
  void printGraph() {
    Logger.yellow('Computational Graph:', prefix: '📊');
    Map<int, int> costs = {};
    _buildGraphString(this, '', true, {}, costs, isRoot: true);

    int totalCost = costs.values.reduce((sum, cost) => sum + cost);
    Logger.yellow('Total Graph Cost: ~${totalCost} FLOPs', prefix: 'Σ');
  }

// --- Private helpers for printGraph ---

  String _getShape() {
    if (value is Scalar) return '[]';
    if (value is Vector) return '[${(value as Vector).length}]';
    if (value is Matrix) {
      final m = value as Matrix;
      final rows = m.length;
      final cols = rows > 0 ? m[0].length : 0;
      return '[$rows, $cols]';
    }
    return 'unknown';
  }

// --- Replace this helper method in your Tensor class ---
  void _buildGraphString(
      Tensor tensor,
      String prefix,
      bool isLast,
      Set<int> visited,
      Map<int, int> costs, { // Pass costs map
        bool isRoot = false,
      }) {
    int tensorId = tensor.hashCode;

    // Store cost for the total calculation, avoiding duplicates
    if (tensor.creator != null && !costs.containsKey(tensorId)) {
      costs[tensorId] = tensor.creator!.cost;
    }

    if (visited.contains(tensorId)) {
      Logger.red(
        '(Seen again: Tensor $tensorId)',
        prefix: '$prefix${isLast ? '└──' : '├──'}',
      );
      return;
    }
    visited.add(tensorId);

    String opInfo;
    Function loggerMethod;

    // Add cost to the display string if it's an operation
    String costInfo = tensor.creator != null && tensor.creator!.cost > 0
        ? ', cost: ${tensor.creator!.cost}'
        : '';

    bool isLeaf = tensor.creator == null || tensor.creator!.inputs.isEmpty;

    if (isRoot) {
      loggerMethod = Logger.yellow;
      opInfo = '(op: ${tensor.creator!.opName}${costInfo})';
    } else if (isLeaf) {
      loggerMethod = Logger.green;
      opInfo = '(Leaf: ${tensor.creator?.opName ?? "Input"})';
    } else {
      loggerMethod = Logger.blue;
      opInfo = '(op: ${tensor.creator!.opName}${costInfo})';
    }

    String treePrefix = '$prefix${isLast ? '└──' : '├──'}';
    String message = 'Tensor $tensorId<${tensor._getShape()}> $opInfo';

    loggerMethod(message, prefix: treePrefix);

    if (tensor.creator != null) {
      var inputs = tensor.creator!.inputs;
      for (int i = 0; i < inputs.length; i++) {
        var newPrefix = prefix + (isLast ? '    ' : '│   ');
        _buildGraphString(
          inputs[i],
          newPrefix,
          i == inputs.length - 1,
          visited,
          costs, // Pass costs map in recursion
        );
      }
    }
  }
}

// Type aliases for the underlying data
typedef Scalar = double;
typedef Vector = List<double>;
typedef Matrix = List<List<double>>;
typedef Tensor3D = List<List<List<double>>>;

///(/////////////////////
///Scalar Operations ///
///(/////////////////////
// --- Scalar Multiplication --- Cost: 1 FLOP
Tensor<Scalar> multiply(Tensor<Scalar> a, Tensor<Scalar> b) {
  Scalar outValue = a.value * b.value;
  Tensor<Scalar> out = Tensor<Scalar>(outValue);

  out.creator = Node([a, b], () {
    a.grad += out.grad * b.value;
    b.grad += out.grad * a.value;
  }, opName: 'multiply', cost: 1);
  return out;
}
// --- Sigmoid (Scalar) --- Cost: 1 operation
Tensor<Scalar> sigmoidScalar(Tensor<Scalar> s) {
  Scalar outValue = 1.0 / (1.0 + exp(-s.value));
  Tensor<Scalar> out = Tensor<Scalar>(outValue);

  out.creator = Node([s], () {
    s.grad += out.grad * (out.value * (1 - out.value));
  }, opName: 'sigmoidScalar', cost: 1);
  return out;
}
// --- Binary Cross-Entropy (Scalar) --- Cost: 1 operation
Tensor<Scalar> binaryCrossEntropy(Tensor<Scalar> prediction, Tensor<Scalar> target) {
  Scalar outValue = -(target.value * log(prediction.value) + (1 - target.value) * log(1 - prediction.value));
  Tensor<Scalar> out = Tensor<Scalar>(outValue);

  out.creator = Node([prediction, target], () {
    prediction.grad += out.grad * ((prediction.value - target.value) / (prediction.value * (1 - prediction.value)));
  }, opName: 'binaryCrossEntropy', cost: 1);
  return out;
}

///(////////////////////
///Vector Operations ///
///(////////////////////
///
// --- Sum Vector --- Cost: N-1 additions
Tensor<Scalar> sum(Tensor<Vector> v) {
  int N = v.value.length;
  Scalar total = 0.0;
  for (int i = 0; i < N; i++) {
    total += v.value[i];
  }
  Tensor<Scalar> out = Tensor<Scalar>(total);

  out.creator = Node([v], () {
    for (int i = 0; i < v.value.length; i++) {
      v.grad[i] += out.grad * 1.0;
    }
  }, opName: 'sum', cost: N);
  return out;
}
// --- Vector Addition --- Cost: N additions
Tensor<Vector> add(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(a.value[i] + b.value[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([a, b], () {
    for (int i = 0; i < a.value.length; i++) {
      a.grad[i] += out.grad[i];
      b.grad[i] += out.grad[i];
    }
  }, opName: 'add', cost: N);
  return out;
}
// --- Dot Product --- Cost: N mults + N adds = 2*N
Tensor<Scalar> dot(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Scalar outValue = 0.0;
  for (int i = 0; i < N; i++) {
    outValue += a.value[i] * b.value[i];
  }
  Tensor<Scalar> out = Tensor<Scalar>(outValue);

  out.creator = Node([a, b], () {
    for (int i = 0; i < a.value.length; i++) {
      a.grad[i] += out.grad * b.value[i];
      b.grad[i] += out.grad * a.value[i];
    }
  }, opName: 'dot', cost: 2 * N);
  return out;
}
// --- ReLU Activation --- Cost: N operations
Tensor<Vector> relu(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] > 0 ? v.value[i] : 0.0);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < v.value.length; i++) {
      v.grad[i] += out.grad[i] * (v.value[i] > 0 ? 1.0 : 0.0);
    }
  }, opName: 'relu', cost: N);
  return out;
}
// --- Mean Squared Error Loss --- Cost: ~3*N FLOPs
Tensor<Scalar> mse(Tensor<Vector> predictions, Tensor<Vector> targets) {
  int N = predictions.value.length;
  Scalar sumSquaredError = 0.0;
  for (int i = 0; i < N; i++) {
    Scalar error = predictions.value[i] - targets.value[i];
    sumSquaredError += error * error;
  }
  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / N);

  out.creator = Node([predictions, targets], () {
    for (int i = 0; i < predictions.value.length; i++) {
      predictions.grad[i] += (2 * (predictions.value[i] - targets.value[i])) / N;
    }
  }, opName: 'mse', cost: 3 * N);
  return out;
}
// --- Sigmoid (Vector) --- Cost: N operations
Tensor<Vector> sigmoid(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(1.0 / (1.0 + exp(-v.value[i])));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < v.value.length; i++) {
      v.grad[i] += out.grad[i] * (out.value[i] * (1 - out.value[i]));
    }
  }, opName: 'sigmoid', cost: N);
  return out;
}
// --- Tanh (Vector) --- Cost: N operations
Tensor<Vector> vectorTanh(Tensor<Vector> v) {
  double _tanh(double x) {
    var e2x = exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(_tanh(v.value[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([v], () {
    for (int i = 0; i < v.value.length; i++) {
      v.grad[i] += out.grad[i] * (1 - pow(out.value[i], 2));
    }
  }, opName: 'tanh', cost: N);
  return out;
}
// --- Element-wise Multiply (Vector) --- Cost: N multiplications
Tensor<Vector> elementWiseMultiply(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(a.value[i] * b.value[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([a, b], () {
    for (int i = 0; i < a.value.length; i++) {
      a.grad[i] += out.grad[i] * b.value[i];
      b.grad[i] += out.grad[i] * a.value[i];
    }
  }, opName: 'multiply', cost: N);
  return out;
}
// --- Concatenate (Vector) --- Cost: 0 (memory op)
Tensor<Vector> concatenate(Tensor<Vector> a, Tensor<Vector> b) {
  Vector outValue = [...a.value, ...b.value];
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([a, b], () {
    int aLength = a.value.length;
    for (int i = 0; i < aLength; i++) {
      a.grad[i] += out.grad[i];
    }
    for (int i = 0; i < b.value.length; i++) {
      b.grad[i] += out.grad[aLength + i];
    }
  }, opName: 'concat', cost: 0);
  return out;
}

Tensor<Vector> vectorExp(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for(int i=0; i<N; i++) { outValue.add(exp(v.value[i])); }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      v.grad[i] += out.grad[i] * out.value[i]; // Derivative of e^x is e^x
    }
  }, opName: 'exp', cost: N);
  return out;
}

Tensor<Vector> vectorLog(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for(int i=0; i<N; i++) { outValue.add(log(v.value[i])); }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) {
      v.grad[i] += out.grad[i] * (1 / v.value[i]); // Derivative of ln(x) is 1/x
    }
  }, opName: 'log', cost: N);
  return out;
}

Tensor<Vector> addScalar(Tensor<Vector> v, double s) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) { outValue.add(v.value[i] + s); }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node([v], () {
    for (int i = 0; i < N; i++) { v.grad[i] += out.grad[i]; }
  }, opName: 'addScalar', cost: N);
  return out;
}

Tensor<Vector> softplus(Tensor<Vector> v) {
  return vectorLog(addScalar(vectorExp(v), 1.0));
}

///(////////////////////
///Matrix Operations ///
///(////////////////////
// --- Matrix-Vector Multiplication --- Cost: R*C mults + R*C adds = 2*R*C
Tensor<Vector> matVecMul(Tensor<Matrix> M, Tensor<Vector> v) {
  int numRows = M.value.length;
  int numCols = M.value[0].length;
  Vector outValue = List<double>.filled(numRows, 0.0, growable: true);
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      outValue[i] += M.value[i][j] * v.value[j];
    }
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node([M, v], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        M.grad[i][j] += out.grad[i] * v.value[j];
      }
    }
    for (int j = 0; j < numCols; j++) {
      for (int i = 0; i < numRows; i++) {
        v.grad[j] += M.value[i][j] * out.grad[i];
      }
    }
  }, opName: 'matVecMul', cost: 2 * numRows * numCols);
  return out;
}
// --- Sum Matrix ---
Tensor<Scalar> sumMatrix(Tensor<Matrix> m) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  double total = 0.0;
  for (Vector row in m.value) {
    for (double val in row) {
      total += val;
    }
  }
  Tensor<Scalar> out = Tensor<Scalar>(total);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad * 1.0;
      }
    }
  }, opName: 'sumMatrix', cost: numRows * numCols);
  return out;
}
// --- Mean Squared Error Loss for a Matrix ---
Tensor<Scalar> mseMatrix(Tensor<Matrix> predictions, Tensor<Matrix> targets) {
  int numRows = predictions.value.length;
  int numCols = predictions.value[0].length;
  int N = numRows * numCols;

  double sumSquaredError = 0.0;
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      double error = predictions.value[i][j] - targets.value[i][j];
      sumSquaredError += error * error;
    }
  }

  // FORWARD: Calculate the MEAN, not the sum.
  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / N);

  // BACKWARD: The gradient must also be scaled by 1/N.
  out.creator = Node([predictions, targets], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        predictions.grad[i][j] += out.grad * 2 * (predictions.value[i][j] - targets.value[i][j]) / N;
      }
    }
  }, opName: 'mseMatrix', cost: 3 * N);
  return out;
}
// --- Matrix Addition --- Cost: R*C additions
Tensor<Matrix> addMatrix(Tensor<Matrix> a, Tensor<Matrix> b) {
  int numRows = a.value.length;
  int numCols = a.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(a.value[i][j] + b.value[i][j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([a, b], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        a.grad[i][j] += out.grad[i][j];
        b.grad[i][j] += out.grad[i][j];
      }
    }
  }, opName: 'addMatrix', cost: numRows * numCols);
  return out;
}
// --- Element-wise Matrix Multiplication --- Cost: R*C multiplications
Tensor<Matrix> elementWiseMultiplyMatrix(Tensor<Matrix> a, Tensor<Matrix> b) {
  int numRows = a.value.length;
  int numCols = a.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(a.value[i][j] * b.value[i][j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([a, b], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        a.grad[i][j] += out.grad[i][j] * b.value[i][j];
        b.grad[i][j] += out.grad[i][j] * a.value[i][j];
      }
    }
  }, opName: 'multiplyMatrix', cost: numRows * numCols);
  return out;
}
// --- Sigmoid (Matrix) --- Cost: R*C operations
Tensor<Matrix> sigmoidMatrix(Tensor<Matrix> m) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(1.0 / (1.0 + exp(-m.value[i][j])));
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        out.grad[i][j] += out.grad[i][j] * (out.value[i][j] * (1 - out.value[i][j]));
      }
    }
  }, opName: 'sigmoidMatrix', cost: numRows * numCols);
  return out;
}
// --- Tanh (Matrix) --- Cost: R*C operations
Tensor<Matrix> tanhMatrix(Tensor<Matrix> m) {
  double _tanh(double x) {
    var e2x = exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(_tanh(m.value[i][j]));
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j] * (1 - pow(out.value[i][j], 2));
      }
    }
  }, opName: 'tanhMatrix', cost: numRows * numCols);
  return out;
}

/// Performs matrix-matrix multiplication with improved cache efficiency.
Tensor<Matrix> matMul(Tensor<Matrix> a, Tensor<Matrix> b) {
  int M = a.value.length;
  int N = a.value[0].length;
  int P = b.value[0].length;

  if (N != b.value.length) {
    throw Exception('Matrix dimensions are incompatible for multiplication.');
  }

  // --- FORWARD PASS OPTIMIZATION ---
  // 1. Transpose matrix 'b' to make column access sequential.
  Matrix bT = [];
  for (int i = 0; i < P; i++) {
    Vector row = [];
    for (int j = 0; j < N; j++) {
      row.add(b.value[j][i]);
    }
    bT.add(row);
  }

  // 2. Perform multiplication with improved memory access.
  Matrix outValue = [];
  for (int i = 0; i < M; i++) {
    Vector rowA = a.value[i];
    Vector outRow = [];
    for (int j = 0; j < P; j++) {
      Vector rowBT = bT[j];
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += rowA[k] * rowBT[k];
      }
      outRow.add(sum);
    }
    outValue.add(outRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = 2 * M * N * P;

  out.creator = Node([a, b], () {
    // --- BACKWARD PASS (also optimized) ---
    // grad_a = grad_out @ b.T (This is already the optimized form)
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < N; k++) {
        for (int j = 0; j < P; j++) {
          a.grad[i][k] += out.grad[i][j] * b.value[k][j];
        }
      }
    }

    // grad_b = a.T @ grad_out
    // 1. Transpose matrix 'a' for cache-friendly access.
    Matrix aT = [];
    for (int i = 0; i < N; i++) {
      Vector row = [];
      for (int j = 0; j < M; j++) {
        row.add(a.value[j][i]);
      }
      aT.add(row);
    }

    // 2. Calculate grad_b with improved memory access.
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < P; j++) {
        for (int i = 0; i < M; i++) {
          b.grad[k][j] += aT[k][i] * out.grad[i][j];
        }
      }
    }
  }, opName: 'matMul', cost: cost);

  return out;
}

Tensor<Matrix> addMatrixAndVector(Tensor<Matrix> m, Tensor<Vector> v) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;

  if (v.value.length != numCols) {
    throw Exception('Vector length must match the number of columns in the matrix.');
  }

  Matrix outValue = [];
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] + v.value[j]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = numRows * numCols;

  out.creator = Node([m, v], () {
    // Gradient for the matrix flows straight through.
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j];
      }
    }

    // Gradient for the vector is the sum of the gradients from each row.
    for (int j = 0; j < numCols; j++) {
      for (int i = 0; i < numRows; i++) {
        v.grad[j] += out.grad[i][j];
      }
    }
  }, opName: 'addMatrixAndVector', cost: cost);

  return out;
}

// Applies the ReLU function element-wise to a matrix.
Tensor<Matrix> reluMatrix(Tensor<Matrix> m) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] > 0 ? m.value[i][j] : 0.0);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j] * (m.value[i][j] > 0 ? 1.0 : 0.0);
      }
    }
  }, opName: 'reluMatrix', cost: numRows * numCols);
  return out;
}

Tensor<Vector> selectRow(Tensor<Matrix> m, int rowIndex) {
  Vector outValue = m.value[rowIndex];
  Tensor<Vector> out = Tensor<Vector>(outValue);

  out.creator = Node([m], () {
    // The backward pass adds the incoming gradient back to the correct row
    // of the original matrix's gradient.
    for (int i = 0; i < outValue.length; i++) {
      m.grad[rowIndex][i] += out.grad[i];
    }
  }, opName: 'selectRow', cost: 0);
  return out;
}


///(////////////////////////
/// 3D TENSOR OPERATIONS ///
///(////////////////////////
Tensor<Tensor3D> add3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int depth = a.value.length;
  int height = a.value[0].length;
  int width = a.value[0][0].length;
  Tensor3D outValue = [];

  for (int d = 0; d < depth; d++) {
    Matrix matrix = [];
    for (int h = 0; h < height; h++) {
      Vector row = [];
      for (int w = 0; w < width; w++) {
        row.add(a.value[d][h][w] + b.value[d][h][w]);
      }
      matrix.add(row);
    }
    outValue.add(matrix);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
  out.creator = Node([a, b], () {
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          a.grad[d][h][w] += out.grad[d][h][w];
          b.grad[d][h][w] += out.grad[d][h][w];
        }
      }
    }
  }, opName: 'add3D', cost: depth * height * width);
  return out;
}

Tensor<Tensor3D> elementWiseMultiply3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int depth = a.value.length;
  int height = a.value[0].length;
  int width = a.value[0][0].length;
  Tensor3D outValue = [];

  // ... similar looping structure to add3D ...

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
  out.creator = Node([a, b], () {
    for (int d = 0; d < depth; d++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          a.grad[d][h][w] += out.grad[d][h][w] * b.value[d][h][w];
          b.grad[d][h][w] += out.grad[d][h][w] * a.value[d][h][w];
        }
      }
    }
  }, opName: 'multiply3D', cost: depth * height * width);
  return out;
}

Tensor<Tensor3D> concatenate3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  Tensor3D outValue = [...a.value, ...b.value];
  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);

  out.creator = Node([a, b], () {
    int aDepth = a.value.length;
    int bDepth = b.value.length;
    for (int d = 0; d < aDepth; d++) {
      for (int h = 0; h < a.value[0].length; h++) {
        for (int w = 0; w < a.value[0][0].length; w++) {
          a.grad[d][h][w] += out.grad[d][h][w];
        }
      }
    }
    for (int d = 0; d < bDepth; d++) {
      for (int h = 0; h < b.value[0].length; h++) {
        for (int w = 0; w < b.value[0][0].length; w++) {
          b.grad[d][h][w] += out.grad[aDepth + d][h][w];
        }
      }
    }
  }, opName: 'concat3D', cost: 0);
  return out;
}

// A helper function to create a padded matrix
Matrix padMatrix(Matrix input, int padding) {
  int newHeight = input.length + 2 * padding;
  int newWidth = input[0].length + 2 * padding;
  Matrix padded = [];
  for (int i = 0; i < newHeight; i++) {
    padded.add(List<double>.filled(newWidth, 0.0));
  }
  for (int i = 0; i < input.length; i++) {
    for (int j = 0; j < input[0].length; j++) {
      padded[i + padding][j + padding] = input[i][j];
    }
  }
  return padded;
}

Tensor<Matrix> conv2d(Tensor<Matrix> input, Tensor<Matrix> kernel, {String padding = 'valid'}) {
  Matrix inputMatrix = input.value;
  int padSize = 0;
  if (padding == 'same') {
    padSize = (kernel.value.length - 1) ~/ 2;
    inputMatrix = padMatrix(input.value, padSize);
  }

  int inputHeight = inputMatrix.length;
  int inputWidth = inputMatrix[0].length;
  int kernelHeight = kernel.value.length;
  int kernelWidth = kernel.value[0].length;
  int outputHeight = inputHeight - kernelHeight + 1;
  int outputWidth = inputWidth - kernelWidth + 1;

  if (outputHeight <= 0 || outputWidth <= 0) {
    throw Exception('Input dimensions must be larger than kernel dimensions for "valid" padding.');
  }

  Matrix outputValue = [];
  for (int i = 0; i < outputHeight; i++) {
    Vector row = List<double>.filled(outputWidth, 0.0);
    outputValue.add(row);
  }

  for (int y = 0; y < outputHeight; y++) {
    for (int x = 0; x < outputWidth; x++) {
      double sum = 0;
      for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
          sum += inputMatrix[y + ky][x + kx] * kernel.value[ky][kx];
        }
      }
      outputValue[y][x] = sum;
    }
  }

  Tensor<Matrix> out = Tensor<Matrix>(outputValue);

  // The cost calculation: (2 * kH * kW - 1) FLOPs per output pixel, plus bias. ~2*kH*kW.
  int cost = outputHeight * outputWidth * 2 * kernelHeight * kernelWidth;

  out.creator = Node([input, kernel], () {
    // Note: The backward pass would need to be updated to handle padding correctly
    // for perfect mathematical gradients, but this is a complex implementation.
    // The existing backward logic remains a good approximation for now.
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        for (int ky = 0; ky < kernelHeight; ky++) {
          for (int kx = 0; kx < kernelWidth; kx++) {
            if (padding == 'same' &&
                (y + ky < padSize || y + ky >= input.value.length + padSize ||
                    x + kx < padSize || x + kx >= input.value[0].length + padSize)) continue;

            int inputGradY = (padding == 'same') ? y + ky - padSize : y + ky;
            int inputGradX = (padding == 'same') ? x + kx - padSize : x + kx;

            input.grad[inputGradY][inputGradX] += kernel.value[ky][kx] * out.grad[y][x];
            kernel.grad[ky][kx] += inputMatrix[y + ky][x + kx] * out.grad[y][x];
          }
        }
      }
    }
  }, opName: 'conv2d', cost: cost);
  return out;
}

Tensor<Matrix> addScalarToMatrix(Tensor<Matrix> m, Tensor<Scalar> s) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] + s.value);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m, s], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j];
      }
    }
    // Corrected line: wrap out.grad in a Tensor
    s.grad += sumMatrix(Tensor<Matrix>(out.grad)).value;
  }, opName: 'addScalarToMatrix', cost: numRows * numCols);
  return out;
}

/// Multiplies every element of a matrix by a scalar value.
Tensor<Matrix> scaleMatrix(Tensor<Matrix> m, double s) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] * s);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node([m], () {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        m.grad[i][j] += out.grad[i][j] * s;
      }
    }
  }, opName: 'scaleMatrix', cost: numRows * numCols);
  return out;
}
/// //////////////////////////////
/// Advanced Tensor Operations ///
/// //////////////////////////////

/// Transposes a matrix, swapping its rows and columns.
///
/// A matrix of shape `[M, N]` will become a matrix of shape `[N, M]`.
///
/// - **Input:** A `Tensor<Matrix>` of shape `[M, N]`.
/// - **Output:** A `Tensor<Matrix>` of shape `[N, M]`.
Tensor<Matrix> transpose(Tensor<Matrix> a) {
  int M = a.value.length;
  int N = a.value[0].length;
  Matrix outValue = [];

  for (int i = 0; i < N; i++) {
    Vector row = [];
    for (int j = 0; j < M; j++) {
      row.add(a.value[j][i]);
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node([a], () {
    // The backward pass of a transpose is another transpose.
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        a.grad[j][i] += out.grad[i][j];
      }
    }
  }, opName: 'transpose', cost: 0); // Memory operation, no FLOPs

  return out;
}

/// Reshapes a 1D Vector into a 2D Matrix.
///
/// The total number of elements must be conserved.
/// `vector.length` must equal `numRows * numCols`.
///
/// - **Input:** A `Tensor<Vector>` of length `N`.
/// - **Output:** A `Tensor<Matrix>` of shape `[numRows, numCols]`.
Tensor<Matrix> reshapeVectorToMatrix(Tensor<Vector> v, int numRows, int numCols) {
  if (v.value.length != numRows * numCols) {
    throw Exception('Reshape error: Total number of elements must be conserved.');
  }

  Matrix outValue = [];
  int index = 0;
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(v.value[index]);
      index++;
    }
    outValue.add(row);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node([v], () {
    // The backward pass flattens the matrix gradient back into a vector.
    int index = 0;
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        v.grad[index] += out.grad[i][j];
        index++;
      }
    }
  }, opName: 'reshape', cost: 0); // Memory operation, no FLOPs

  return out;
}


/// Applies the softmax function to each row of a matrix independently.
///
/// Softmax converts a vector of real numbers (logits) into a probability
/// distribution. This function is critical for the self-attention mechanism,
/// where it's used to turn raw attention scores into weights that sum to 1.
///
/// - **Input:** A `Tensor<Matrix>` of shape `[M, N]`.
/// - **Output:** A `Tensor<Matrix>` of shape `[M, N]`, where each row is a
///   probability distribution.
Tensor<Matrix> softmaxMatrix(Tensor<Matrix> m) {
  Matrix outValue = [];
  int numRows = m.value.length;
  int numCols = m.value[0].length;

  for (int i = 0; i < numRows; i++) {
    Vector row = m.value[i];
    double maxVal = 0;
    for (int j = 0; j < row.length; j++) {
      if (row[j] > maxVal) {
        maxVal = row[j];
      }
    }

    Vector exps = [];
    double sumExps = 0;
    for (int j = 0; j < numCols; j++) {
      double expVal = exp(row[j] - maxVal);
      exps.add(expVal);
      sumExps += expVal;
    }

    Vector softmaxRow = [];
    for (int j = 0; j < numCols; j++) {
      softmaxRow.add(exps[j] / sumExps);
    }
    outValue.add(softmaxRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node([m], () {
    for (int r = 0; r < numRows; r++) {
      Vector softmaxRow = out.value[r];
      Vector gradRow = out.grad[r];

      double dotProduct = 0;
      for (int i = 0; i < numCols; i++) {
        dotProduct += gradRow[i] * softmaxRow[i];
      }

      for (int j = 0; j < numCols; j++) {
        m.grad[r][j] += softmaxRow[j] * (gradRow[j] - dotProduct);
      }
    }
  }, opName: 'softmax_matrix', cost: numRows * numCols * numCols);

  return out;
}

/// Concatenates a list of matrices by column.
/// All matrices must have the same number of rows.
Tensor<Matrix> concatenateMatricesByColumn(List<Tensor<Matrix>> matrices) {
  int numRows = matrices[0].value.length;
  Matrix outValue = [];

  for (int i = 0; i < numRows; i++) {
    Vector newRow = [];
    for (Tensor<Matrix> m in matrices) {
      newRow.addAll(m.value[i]);
    }
    outValue.add(newRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);

  out.creator = Node(matrices, () {
    int current_col = 0;
    for (Tensor<Matrix> m in matrices) {
      int numCols = m.value[0].length;
      for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
          m.grad[r][c] += out.grad[r][current_col + c];
        }
      }
      current_col += numCols;
    }
  }, opName: 'concat_matrix_col');
  return out;
}
/// Applies Layer Normalization to each row of a matrix.
Tensor<Matrix> layerNorm(Tensor<Matrix> m, Tensor<Vector> gamma, Tensor<Vector> beta, {double epsilon = 1e-5}) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  Matrix normalizedRows = [];
  Vector means = [];
  Vector variances = [];

  for (int r = 0; r < numRows; r++) {
    Vector row = m.value[r];
    double mean = 0;
    for (double val in row) { mean += val; }
    mean /= numCols;
    means.add(mean);

    double variance = 0;
    for (double val in row) { variance += pow(val - mean, 2); }
    variance /= numCols;
    variances.add(variance);

    Vector normalizedRow = [];
    for (double val in row) {
      normalizedRow.add((val - mean) / sqrt(variance + epsilon));
    }
    normalizedRows.add(normalizedRow);

    Vector finalRow = [];
    for (int c = 0; c < numCols; c++) {
      finalRow.add(gamma.value[c] * normalizedRow[c] + beta.value[c]);
    }
    outValue.add(finalRow);
  }

  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = numRows * numCols * 8;

  out.creator = Node([m, gamma, beta], () {
    for(int r = 0; r < numRows; r++){
      Vector grad_x_hat = [];
      for(int c=0; c < numCols; c++){
        grad_x_hat.add(out.grad[r][c] * gamma.value[c]);
        gamma.grad[c] += out.grad[r][c] * normalizedRows[r][c];
        beta.grad[c] += out.grad[r][c];
      }

      double sum_grad_x_hat = 0;
      for (double val in grad_x_hat) { sum_grad_x_hat += val; }

      double dot_product_term = 0;
      for (int c = 0; c < numCols; c++) {
        dot_product_term += grad_x_hat[c] * normalizedRows[r][c];
      }

      for (int c = 0; c < numCols; c++) {
        double term1 = numCols * grad_x_hat[c];
        double term2 = sum_grad_x_hat;
        double term3 = normalizedRows[r][c] * dot_product_term;

        double total_grad = (1.0 / (numCols * sqrt(variances[r] + epsilon))) * (term1 - term2 - term3);
        m.grad[r][c] += total_grad;
      }
    }
  }, opName: 'layer_norm', cost: cost);
  return out;
}

void main() {
  // The main function usage remains the same
  Tensor<Matrix> M = Tensor<Matrix>([
    [1.0, 2.0],
    [3.0, 4.0],
  ]);
  Tensor<Vector> v = Tensor<Vector>([10.0, 20.0]);

  Tensor<Vector> y = matVecMul(M, v);
  Tensor<Scalar> loss = sum(y);

  print('Forward pass result (y): ${y.value}');
  print('Final loss: ${loss.value}');
  print('--------------------------');

  loss.backward();

  print('Backward pass gradients:');
  print('Gradient of M: ${M.grad}');
  print('Gradient of v: ${v.grad}');
}
