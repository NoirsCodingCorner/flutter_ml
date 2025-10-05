import 'dart:math';

import '../activationFunctions/gelu.dart';
import '../diagnosysTools/logger.dart';
import '../flutter_ml.dart';

class Node {
  final List<Tensor> inputs;
  final Function backwardFn;
  final String opName;
  final int cost;
  final Map<String, dynamic> extraParams; // <-- Add this field

  Node(
    this.inputs,
    this.backwardFn, {
    this.opName = 'op',
    this.cost = 0,
    this.extraParams = const {},
  });
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

  /// Prints a color-coded, visual representation of the computational graph,
  /// including computational cost and parallelizable dependency levels.
  void printGraph() {
    Logger.yellow('Computational Graph:', prefix: '📊');

    // 1. Perform a topological sort to get all nodes in dependency order.
    List<Node> topo = [];
    Set<Node> visited = {};
    void buildTopo(Node? node) {
      if (node == null || visited.contains(node)) return;
      visited.add(node);
      for (Tensor inputTensor in node.inputs) {
        buildTopo(inputTensor.creator);
      }
      topo.add(node);
    }

    buildTopo(creator);

    // 2. Calculate the dependency level for each node.
    Map<Node, int> levels = {};
    for (Node node in topo) {
      int maxInputLevel = -1;
      for (Tensor inputTensor in node.inputs) {
        if (inputTensor.creator != null) {
          int inputLevel = levels[inputTensor.creator!] ?? 0;
          if (inputLevel > maxInputLevel) {
            maxInputLevel = inputLevel;
          }
        }
      }
      levels[node] = maxInputLevel + 1;
    }

    // 3. Recursively print the graph, passing the level information.
    Map<int, int> costs = {};
    _buildGraphString(this, '', true, {}, costs, levels, isRoot: true);

    int totalCost = costs.values.fold(0, (sum, cost) => sum + cost);
    Logger.yellow('Total Graph Cost: ~${totalCost} FLOPs', prefix: 'Σ');
  }

  void _buildGraphString(
    Tensor tensor,
    String prefix,
    bool isLast,
    Set<int> visited,
    Map<int, int> costs,
    Map<Node, int> levels, { // Pass the levels map
    bool isRoot = false,
  }) {
    int tensorId = tensor.hashCode;

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
    String levelInfo = ''; // NEW: To hold the level string
    Function loggerMethod;

    String costInfo =
        tensor.creator != null && tensor.creator!.cost > 0
            ? ', cost: ${tensor.creator!.cost}'
            : '';

    bool isLeaf = tensor.creator == null || tensor.creator!.inputs.isEmpty;

    if (isRoot) {
      loggerMethod = Logger.yellow;
      levelInfo = ' [Lvl: ${levels[tensor.creator!]}]';
      opInfo = '(op: ${tensor.creator!.opName}${costInfo})';
    } else if (isLeaf) {
      loggerMethod = Logger.green;
      levelInfo = ' [Lvl: 0]';
      opInfo = '(Leaf: ${tensor.creator?.opName ?? "Input"})';
    } else {
      loggerMethod = Logger.blue;
      levelInfo = ' [Lvl: ${levels[tensor.creator!]}]';
      opInfo = '(op: ${tensor.creator!.opName}${costInfo})';
    }

    String treePrefix = '$prefix${isLast ? '└──' : '├──'}';
    String message =
        'Tensor $tensorId<${tensor._getShape()}> $opInfo$levelInfo';

    loggerMethod(message, prefix: treePrefix);

    if (tensor.creator != null) {
      List<Tensor> inputs = tensor.creator!.inputs;
      for (int i = 0; i < inputs.length; i++) {
        String newPrefix = prefix + (isLast ? '    ' : '│   ');
        _buildGraphString(
          inputs[i],
          newPrefix,
          i == inputs.length - 1,
          visited,
          costs,
          levels,
        );
      }
    }
  }

  // Add this method to your Tensor class
  void printParallelGraph() {
    Logger.yellow('Parallel Computational Graph:', prefix: '⚡️');

    // 1. Perform a topological sort to get all nodes in dependency order.
    List<Node> topo = [];
    Set<Node> visitedNodes = {};
    void buildTopo(Node? node) {
      if (node == null || visitedNodes.contains(node)) return;
      visitedNodes.add(node);
      for (Tensor inputTensor in node.inputs) {
        buildTopo(inputTensor.creator);
      }
      topo.add(node);
    }

    buildTopo(creator);

    // 2. Calculate the dependency level for each node.
    Map<Node, int> levels = {};
    for (Node node in topo) {
      int maxInputLevel = -1;
      for (Tensor inputTensor in node.inputs) {
        if (inputTensor.creator != null) {
          int inputLevel = levels[inputTensor.creator!] ?? 0;
          if (inputLevel > maxInputLevel) {
            maxInputLevel = inputLevel;
          }
        }
      }
      levels[node] = maxInputLevel + 1;
    }

    // 3. Group nodes by their dependency level.
    Map<int, List<Node>> parallelGroups = {};
    for (Node node in topo) {
      int level = levels[node] ?? 0;
      if (!parallelGroups.containsKey(level)) {
        parallelGroups[level] = [];
      }
      parallelGroups[level]!.add(node);
    }

    // 4. Print the grouped operations level by level.
    List<int> sortedLevels = parallelGroups.keys.toList()..sort();
    int totalCost = 0;

    for (int level in sortedLevels) {
      List<Node> nodesInLevel = parallelGroups[level]!;
      int levelCost = 0;
      for (Node node in nodesInLevel) {
        levelCost += node.cost;
      }
      totalCost += levelCost;

      Logger.cyan(
        '--- Level ${level + 1} (${nodesInLevel.length} parallel ops, cost: $levelCost) ---',
        prefix: '',
      );
      for (Node node in nodesInLevel) {
        print('  - op: ${node.opName}, cost: ${node.cost}');
      }
    }

    Logger.yellow('Total Graph Cost: ~${totalCost} FLOPs', prefix: 'Σ');
  }
}

// Type Aliases
typedef Scalar = double;
typedef Vector = List<double>;
typedef Matrix = List<List<double>>;
typedef Tensor3D = List<List<List<double>>>;

////////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS (No Graph Connection)
////////////////////////////////////////////////////////////////////////////////

// Note: This function doesn't create a computational graph node, so it's fine as is.
Matrix padMatrix(Matrix input, int padding) {
  int newHeight = input.length + 2 * padding;
  int newWidth = input[0].length + 2 * padding;
  Matrix padded = [];
  for (int i = 0; i < newHeight; i++) {
    Vector row = [];
    for (int j = 0; j < newWidth; j++) {
      row.add(0.0);
    }
    padded.add(row);
  }
  for (int i = 0; i < input.length; i++) {
    for (int j = 0; j < input[0].length; j++) {
      padded[i + padding][j + padding] = input[i][j];
    }
  }
  return padded;
}

////////////////////////////////////////////////////////////////////////////////
// SCALAR (0D) OPERATIONS
////////////////////////////////////////////////////////////////////////////////

Tensor<Scalar> multiply(Tensor<Scalar> a, Tensor<Scalar> b) {
  Scalar outValue = a.value * b.value;
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node(
    [a, b],
        () {
      a.grad += out.grad * b.value;
      b.grad += out.grad * a.value;
    },
    opName: 'multiply_scalar', // <-- Changed for clarity and to fix the bug
    cost: 1,
  );
  return out;
}

Tensor<Scalar> sigmoidScalar(Tensor<Scalar> s) {
  Scalar outValue = 1.0 / (1.0 + exp(-s.value));
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node(
    [s],
        () {
      s.grad += out.grad * (out.value * (1 - out.value));
    },
    opName: 'sigmoidScalar',
    cost: 1,
  );
  return out;
}

Tensor<Scalar> binaryCrossEntropy(
    Tensor<Scalar> prediction,
    Tensor<Scalar> target,
    ) {
  Scalar outValue =
  -(target.value * log(prediction.value) +
      (1 - target.value) * log(1 - prediction.value));
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node(
    [prediction, target],
        () {
      prediction.grad +=
          out.grad *
              ((prediction.value - target.value) /
                  (prediction.value * (1 - prediction.value)));
    },
    opName: 'binaryCrossEntropy',
    cost: 1,
  );
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// VECTOR (1D) OPERATIONS
////////////////////////////////////////////////////////////////////////////////

// Assuming your Tensor, Node, and type aliases are defined.

Tensor<Vector> add(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(a.value[i] + b.value[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < a.value.length; i++) {
        a.grad[i] += out.grad[i];
        b.grad[i] += out.grad[i];
      }
    },
    opName: 'add_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Vector> addScalar(Tensor<Vector> v, double s) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] + s);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i++) {
        v.grad[i] += out.grad[i];
      }
    },
    opName: 'addScalar_vector', // <-- Renamed for clarity
    extraParams: {'s': s},    // <-- CRITICAL: Storing the non-Tensor parameter
    cost: N,
  );
  return out;
}

Tensor<Vector> concatenate(Tensor<Vector> a, Tensor<Vector> b) {
  Vector outValue = [...a.value, ...b.value];
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      int aLength = a.value.length;
      for (int i = 0; i < aLength; i++) {
        a.grad[i] += out.grad[i];
      }
      for (int i = 0; i < b.value.length; i++) {
        b.grad[i] += out.grad[aLength + i];
      }
    },
    opName: 'concat_vector', // <-- Renamed for clarity
    cost: 0,
  );
  return out;
}

Tensor<Scalar> dot(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Scalar outValue = 0.0;
  for (int i = 0; i < N; i++) {
    outValue += a.value[i] * b.value[i];
  }
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < a.value.length; i++) {
        a.grad[i] += out.grad * b.value[i];
        b.grad[i] += out.grad * a.value[i];
      }
    },
    opName: 'dot', // This name is already unique
    cost: 2 * N,
  );
  return out;
}

Tensor<Vector> elementWiseMultiply(Tensor<Vector> a, Tensor<Vector> b) {
  int N = a.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(a.value[i] * b.value[i]);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < a.value.length; i++) {
        a.grad[i] += out.grad[i] * b.value[i];
        b.grad[i] += out.grad[i] * a.value[i];
      }
    },
    opName: 'multiply_vector', // <-- Changed from ambiguous 'multiply'
    cost: N,
  );
  return out;
}

Tensor<Scalar> mse(Tensor<Vector> predictions, Tensor<Vector> targets) {
  int N = predictions.value.length;
  Scalar sumSquaredError = 0.0;
  for (int i = 0; i < N; i++) {
    Scalar error = predictions.value[i] - targets.value[i];
    sumSquaredError += error * error;
  }
  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / N);
  out.creator = Node(
    [predictions, targets],
        () {
      for (int i = 0; i < predictions.value.length; i++) {
        predictions.grad[i] +=
            (2 * (predictions.value[i] - targets.value[i])) / N;
      }
    },
    opName: 'mse_vector', // <-- Renamed for clarity
    cost: 3 * N,
  );
  return out;
}

Tensor<Vector> relu(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(v.value[i] > 0 ? v.value[i] : 0.0);
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < v.value.length; i++) {
        v.grad[i] += out.grad[i] * (v.value[i] > 0 ? 1.0 : 0.0);
      }
    },
    opName: 'relu_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Vector> sigmoid(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(1.0 / (1.0 + exp(-v.value[i])));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < v.value.length; i++) {
        v.grad[i] += out.grad[i] * (out.value[i] * (1 - out.value[i]));
      }
    },
    opName: 'sigmoid_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Scalar> sum(Tensor<Vector> v) {
  int N = v.value.length;
  Scalar total = 0.0;
  for (int i = 0; i < N; i++) {
    total += v.value[i];
  }
  Tensor<Scalar> out = Tensor<Scalar>(total);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < v.value.length; i++) {
        v.grad[i] += out.grad * 1.0;
      }
    },
    opName: 'sum_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorTanh(Tensor<Vector> v) {
  double _tanh(double x) {
    var e2x = exp(2 * x);
    if (e2x.isInfinite) return 1.0;
    return (e2x - 1) / (e2x + 1);
  }

  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(_tanh(v.value[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < v.value.length; i++) {
        v.grad[i] += out.grad[i] * (1 - pow(out.value[i], 2));
      }
    },
    opName: 'tanh_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorExp(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(exp(v.value[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i++) {
        v.grad[i] += out.grad[i] * out.value[i];
      }
    },
    opName: 'exp_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}

Tensor<Vector> vectorLog(Tensor<Vector> v) {
  int N = v.value.length;
  Vector outValue = [];
  for (int i = 0; i < N; i++) {
    outValue.add(log(v.value[i]));
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [v],
        () {
      for (int i = 0; i < N; i++) {
        v.grad[i] += out.grad[i] * (1 / v.value[i]);
      }
    },
    opName: 'log_vector', // <-- Renamed for clarity
    cost: N,
  );
  return out;
}
////////////////////////////////////////////////////////////////////////////////
// MATRIX (2D) OPERATIONS
////////////////////////////////////////////////////////////////////////////////

// Assuming your Tensor, Node, and type aliases are defined.
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
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          a.grad[i][j] += out.grad[i][j];
          b.grad[i][j] += out.grad[i][j];
        }
      }
    },
    opName: 'add_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> addMatrixAndVector(Tensor<Matrix> m, Tensor<Vector> v) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];
  for (int i = 0; i < numRows; i++) {
    Vector row = [];
    for (int j = 0; j < numCols; j++) {
      row.add(m.value[i][j] + v.value[j]);
    }
    outValue.add(row);
  }
  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  out.creator = Node(
    [m, v],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad[i][j];
        }
      }
      for (int j = 0; j < numCols; j++) {
        for (int i = 0; i < numRows; i++) {
          v.grad[j] += out.grad[i][j];
        }
      }
    },
    opName: 'addMatrixAndVector', // This name is already unique
    cost: numRows * numCols,
  );
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
  out.creator = Node(
    [m, s],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad[i][j];
        }
      }
      s.grad += sumMatrix(Tensor<Matrix>(out.grad)).value;
    },
    opName: 'addScalarToMatrix', // This name is already unique
    cost: numRows * numCols,
  );
  return out;
}

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
    int currentCol = 0;
    for (Tensor<Matrix> m in matrices) {
      int numCols = m.value[0].length;
      for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
          m.grad[r][c] += out.grad[r][currentCol + c];
        }
      }
      currentCol += numCols;
    }
  }, opName: 'concat_matrix_col'); // This name is already unique
  return out;
}

Tensor<Matrix> conv2d(
    Tensor<Matrix> input,
    Tensor<Matrix> kernel, {
      String padding = 'valid',
    }) {
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

  Matrix outputValue = [];
  for (int i = 0; i < outputHeight; i++) {
    Vector row = [];
    for (int j = 0; j < outputWidth; j++) {
      row.add(0.0);
    }
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
  int cost = outputHeight * outputWidth * 2 * kernelHeight * kernelWidth;
  out.creator = Node(
    [input, kernel],
        () {
      for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
          for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
              if (padding == 'same' &&
                  (y + ky < padSize ||
                      y + ky >= input.value.length + padSize ||
                      x + kx < padSize ||
                      x + kx >= input.value[0].length + padSize))
                continue;
              int inputGradY = (padding == 'same') ? y + ky - padSize : y + ky;
              int inputGradX = (padding == 'same') ? x + kx - padSize : x + kx;
              input.grad[inputGradY][inputGradX] +=
                  kernel.value[ky][kx] * out.grad[y][x];
              kernel.grad[ky][kx] +=
                  inputMatrix[y + ky][x + kx] * out.grad[y][x];
            }
          }
        }
      }
    },
    opName: 'conv2d',
    extraParams: {'padding': padding}, // <-- CRITICAL: Storing the non-Tensor parameter
    cost: cost,
  );
  return out;
}

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
  out.creator = Node(
    [a, b],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          a.grad[i][j] += out.grad[i][j] * b.value[i][j];
          b.grad[i][j] += out.grad[i][j] * a.value[i][j];
        }
      }
    },
    opName: 'multiply_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}


// Assuming your Tensor, Node, and CudaCaller classes are defined.

/// Performs matrix multiplication, using the GPU for large matrices.
Tensor<Matrix> matMul(Tensor<Matrix> a, Tensor<Matrix> b) {
  int M = a.value.length;
  int N = a.value[0].length;
  int P = b.value[0].length;
  // Note: Add your shape check assertion here if needed

  Matrix outValue; // This will hold the result from either the CPU or GPU path.

  // --- Condition to switch between CPU and GPU ---
  // If either matrix has more than 1000 elements, use the GPU.
  if ((M * N) > 1000 || (N * P) > 1000) {
    // --- 🚀 GPU Path ---
    // 1. Flatten Matrix A from List<List<double>> to List<double>
    List<double> flatA = [];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        flatA.add(a.value[i][j]);
      }
    }

    // 2. Flatten Matrix B
    List<double> flatB = [];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < P; j++) {
        flatB.add(b.value[i][j]);
      }
    }

    // 3. Call the static CUDA method
    List<double> flatC = CudaCaller.matmult(flatA, flatB, M, N, P);

    // 4. Unflatten the flat result back into a List<List<double>>
    outValue = [];
    for (int i = 0; i < M; i++) {
      int startIndex = i * P;
      int endIndex = startIndex + P;
      outValue.add(flatC.sublist(startIndex, endIndex));
    }
  } else {
    // --- 🧠 CPU Path ---
    Matrix bT = [];
    for (int i = 0; i < P; i++) {
      Vector row = [];
      for (int j = 0; j < N; j++) {
        row.add(b.value[j][i]);
      }
      bT.add(row);
    }

    outValue = [];
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
  }

  // --- Common Logic for Both Paths (Backward Pass Setup) ---
  Tensor<Matrix> out = Tensor<Matrix>(outValue);
  int cost = 2 * M * N * P;

  out.creator = Node(
    [a, b],
        () {
      // Create b.T here for the gradient calculation
      Matrix bT = [];
      for (int i = 0; i < P; i++) {
        Vector row = [];
        for (int j = 0; j < N; j++) {
          row.add(b.value[j][i]);
        }
        bT.add(row);
      }

      // grad_a = grad_out @ b.T
      for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++) {
          for (int j = 0; j < P; j++) {
            a.grad[i][k] += out.grad[i][j] * bT[j][k];
          }
        }
      }

      // grad_b = a.T @ grad_out
      Matrix aT = [];
      for (int i = 0; i < N; i++) {
        Vector row = [];
        for (int j = 0; j < M; j++) {
          row.add(a.value[j][i]);
        }
        aT.add(row);
      }
      for (int k = 0; k < N; k++) {
        for (int j = 0; j < P; j++) {
          for (int i = 0; i < M; i++) {
            b.grad[k][j] += aT[k][i] * out.grad[i][j];
          }
        }
      }
    },
    opName: 'matMul', // This name is specific and standard
    cost: cost,
  );
  return out;
}

Tensor<Vector> matVecMul(Tensor<Matrix> M, Tensor<Vector> v) {
  int numRows = M.value.length;
  int numCols = M.value[0].length;
  Vector outValue = [];
  for (int i = 0; i < numRows; i++) {
    outValue.add(0.0);
  }
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; j++) {
      outValue[i] += M.value[i][j] * v.value[j];
    }
  }
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [M, v],
        () {
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
    },
    opName: 'matVecMul', // This name is already unique
    cost: 2 * numRows * numCols,
  );
  return out;
}

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
  Tensor<Scalar> out = Tensor<Scalar>(sumSquaredError / N);
  out.creator = Node(
    [predictions, targets],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          predictions.grad[i][j] +=
              out.grad *
                  2 *
                  (predictions.value[i][j] - targets.value[i][j]) /
                  N;
        }
      }
    },
    opName: 'mse_matrix', // <-- Renamed for clarity
    cost: 3 * N,
  );
  return out;
}

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
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad[i][j] * (m.value[i][j] > 0 ? 1.0 : 0.0);
        }
      }
    },
    opName: 'relu_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> reshapeVectorToMatrix(
    Tensor<Vector> v,
    int numRows,
    int numCols,
    ) {
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
  out.creator = Node(
    [v],
        () {
      int index = 0;
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          v.grad[index] += out.grad[i][j];
          index++;
        }
      }
    },
    opName: 'reshape',
    // <-- CRITICAL: Storing the non-Tensor parameters
    extraParams: {'numRows': numRows, 'numCols': numCols},
    cost: 0,
  );
  return out;
}

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
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad[i][j] * s;
        }
      }
    },
    opName: 'scale_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
    extraParams: {'s': s}, // <-- Storing the non-Tensor parameter
  );
  return out;
}

Tensor<Vector> selectRow(Tensor<Matrix> m, int rowIndex) {
  Vector outValue = m.value[rowIndex];
  Tensor<Vector> out = Tensor<Vector>(outValue);
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < outValue.length; i++) {
        m.grad[rowIndex][i] += out.grad[i];
      }
    },
    opName: 'selectRow',
    // <-- CRITICAL: Storing the non-Tensor parameter
    extraParams: {'rowIndex': rowIndex},
    cost: 0,
  );
  return out;
}

// Assuming your Tensor, Node, and type aliases are defined.

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
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          // BUG FIX: The original gradient calculation was incorrect.
          // It should use m.grad, not out.grad on the left side.
          m.grad[i][j] +=
              out.grad[i][j] * (out.value[i][j] * (1 - out.value[i][j]));
        }
      }
    },
    opName: 'sigmoid_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}

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
  out.creator = Node(
    [m],
        () {
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
    },
    opName: 'softmax_matrix', // This name is already unique
    cost: numRows * numCols * numCols,
  );
  return out;
}

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
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad * 1.0;
        }
      }
    },
    opName: 'sum_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}

Tensor<Matrix> tanhMatrix(Tensor<Matrix> m) {
  double _tanh(double x) {
    var e2x = exp(2 * x);
    if (e2x.isInfinite) return 1.0;
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
  out.creator = Node(
    [m],
        () {
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
          m.grad[i][j] += out.grad[i][j] * (1 - pow(out.value[i][j], 2));
        }
      }
    },
    opName: 'tanh_matrix', // <-- Renamed for clarity
    cost: numRows * numCols,
  );
  return out;
}

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
  out.creator = Node(
    [a],
        () {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
          a.grad[j][i] += out.grad[i][j];
        }
      }
    },
    opName: 'transpose', // This name is already unique
    cost: 0,
  );
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// 3D TENSOR OPERATIONS
////////////////////////////////////////////////////////////////////////////////

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
  out.creator = Node(
    [a, b],
        () {
      for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            a.grad[d][h][w] += out.grad[d][h][w];
            b.grad[d][h][w] += out.grad[d][h][w];
          }
        }
      }
    },
    opName: 'add_3d', // <-- Renamed for clarity
    cost: depth * height * width,
  );
  return out;
}

Tensor<Tensor3D> elementWiseMultiply3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  int depth = a.value.length;
  int height = a.value[0].length;
  int width = a.value[0][0].length;
  Tensor3D outValue = [];

  for (int d = 0; d < depth; d++) {
    Matrix matrix = [];
    for (int h = 0; h < height; h++) {
      Vector row = [];
      for (int w = 0; w < width; w++) {
        row.add(a.value[d][h][w] * b.value[d][h][w]);
      }
      matrix.add(row);
    }
    outValue.add(matrix);
  }

  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);
  out.creator = Node(
    [a, b],
        () {
      for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            a.grad[d][h][w] += out.grad[d][h][w] * b.value[d][h][w];
            b.grad[d][h][w] += out.grad[d][h][w] * a.value[d][h][w];
          }
        }
      }
    },
    opName: 'multiply_3d', // <-- Renamed for clarity
    cost: depth * height * width,
  );
  return out;
}

Tensor<Tensor3D> concatenate3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b) {
  Tensor3D outValue = [...a.value, ...b.value];
  Tensor<Tensor3D> out = Tensor<Tensor3D>(outValue);

  out.creator = Node(
    [a, b],
        () {
      int aDepth = a.value.length;
      for (int d = 0; d < aDepth; d++) {
        for (int h = 0; h < a.value[0].length; h++) {
          for (int w = 0; w < a.value[0][0].length; w++) {
            a.grad[d][h][w] += out.grad[d][h][w];
          }
        }
      }
      int bDepth = b.value.length;
      for (int d = 0; d < bDepth; d++) {
        for (int h = 0; h < b.value[0].length; h++) {
          for (int w = 0; w < b.value[0][0].length; w++) {
            b.grad[d][h][w] += out.grad[aDepth + d][h][w];
          }
        }
      }
    },
    opName: 'concat_3d', // <-- Renamed for clarity
    cost: 0,
  );
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// COMPOSITE OPERATIONS
////////////////////////////////////////////////////////////////////////////////

// This is a composite function; it doesn't create its own Node,
// so it doesn't need changes. The GraphCopier will copy the
// primitive operations it's made of.
Tensor<Vector> softplus(Tensor<Vector> v) {
  return vectorLog(addScalar(vectorExp(v), 1.0));
}

Tensor<Matrix> layerNorm(
    Tensor<Matrix> m,
    Tensor<Vector> gamma,
    Tensor<Vector> beta, {
      double epsilon = 1e-5,
    }) {
  int numRows = m.value.length;
  int numCols = m.value[0].length;
  Matrix outValue = [];

  Matrix normalizedRows = [];
  Vector means = [];
  Vector variances = [];

  for (int r = 0; r < numRows; r++) {
    Vector row = m.value[r];
    double mean = 0;
    for (double val in row) {
      mean += val;
    }
    mean /= numCols;
    means.add(mean);

    double variance = 0;
    for (double val in row) {
      variance += pow(val - mean, 2);
    }
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

  out.creator = Node(
    [m, gamma, beta],
        () {
      for (int r = 0; r < numRows; r++) {
        Vector grad_x_hat = [];
        for (int c = 0; c < numCols; c++) {
          grad_x_hat.add(out.grad[r][c] * gamma.value[c]);
          gamma.grad[c] += out.grad[r][c] * normalizedRows[r][c];
          beta.grad[c] += out.grad[r][c];
        }

        double sum_grad_x_hat = 0;
        for (double val in grad_x_hat) {
          sum_grad_x_hat += val;
        }

        double dot_product_term = 0;
        for (int c = 0; c < numCols; c++) {
          dot_product_term += grad_x_hat[c] * normalizedRows[r][c];
        }

        for (int c = 0; c < numCols; c++) {
          double term1 = numCols * grad_x_hat[c];
          double term2 = sum_grad_x_hat;
          double term3 = normalizedRows[r][c] * dot_product_term;

          double total_grad =
              (1.0 / (numCols * sqrt(variances[r] + epsilon))) *
                  (term1 - term2 - term3);
          m.grad[r][c] += total_grad;
        }
      }
    },
    opName: 'layer_norm', // This name is already unique
    extraParams: {'epsilon': epsilon}, // <-- CRITICAL: Storing the non-Tensor parameter
    cost: cost,
  );
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
