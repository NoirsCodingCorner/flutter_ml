import 'dart:ffi';
import 'package:ffi/ffi.dart';
import '../tensor/tensor_gpu.dart';
import 'dart:typed_data';
import '../logger.dart';

class Node {
  List<Tensor> inputs;
  Function backwardFn;
  String opName;
  int cost;
  Map<String, dynamic> extraParams;

  Node(
      this.inputs,
      this.backwardFn, {
        this.opName = 'op',
        this.cost = 0,
        this.extraParams = const {},
      });
}

class Tensor<T> {
  static int _idCounter = 0;

  late String id;

  late Pointer<Float> dataPtr;
  late Float32List data;
  late Pointer<Float> gradPtr;
  late Float32List grad;

  late List<int> shape;
  Node? creator;

  bool _freed = false;

  // ─────────────────────────────────────────────────────── //
  // Initialization
  // ─────────────────────────────────────────────────────── //

  Tensor(dynamic initialValue, {this.creator}) {
    id = 't_${_idCounter}';
    _idCounter = _idCounter + 1;

    if (initialValue is double) {
      shape = [];
    } else if (initialValue is List<double>) {
      shape = [initialValue.length];
    } else if (initialValue is List<List<double>>) {
      int rows = initialValue.length;
      int cols = rows > 0 ? initialValue[0].length : 0;
      shape = [rows, cols];
    } else if (initialValue is List<List<List<double>>>) {
      int depth  = initialValue.length;
      int height = depth > 0 ? initialValue[0].length : 0;
      int width  = height > 0 ? initialValue[0][0].length : 0;
      shape = [depth, height, width];
    } else {
      throw Exception("Unsupported tensor initialization type.");
    }

    int numElements = _elementCount(shape);

    dataPtr = calloc<Float>(numElements);
    data    = dataPtr.asTypedList(numElements);
    gradPtr = calloc<Float>(numElements);
    grad    = gradPtr.asTypedList(numElements);

    if (initialValue is double) {
      data[0] = initialValue;
    } else if (initialValue is List<double>) {
      for (int i = 0; i < initialValue.length; i = i + 1) {
        data[i] = initialValue[i];
      }
    } else if (initialValue is List<List<double>>) {
      int cols = shape[1];
      for (int i = 0; i < initialValue.length; i = i + 1) {
        for (int j = 0; j < initialValue[i].length; j = j + 1) {
          data[(i * cols) + j] = initialValue[i][j];
        }
      }
    } else if (initialValue is List<List<List<double>>>) {
      int height = shape[1];
      int width  = shape[2];
      for (int d = 0; d < initialValue.length; d = d + 1) {
        for (int h = 0; h < initialValue[d].length; h = h + 1) {
          for (int w = 0; w < initialValue[d][h].length; w = w + 1) {
            data[(d * height * width) + (h * width) + w] = initialValue[d][h][w];
          }
        }
      }
    }
  }

  static int _elementCount(List<int> shape) {
    if (shape.isEmpty) return 1;
    int n = 1;
    for (int i = 0; i < shape.length; i = i + 1) {
      n = n * shape[i];
    }
    return n;
  }

  void free() {
    if (_freed) return;

    calloc.free(dataPtr);
    calloc.free(gradPtr);

    _freed = true;
  }

  // ─────────────────────────────────────────────────────── //
  // Data Unflattening (For CPU Math)
  // ─────────────────────────────────────────────────────── //

  T get value {
    if (shape.isEmpty) return data[0] as T;

    if (shape.length == 1) {
      List<double> vec = [];
      for (int i = 0; i < shape[0]; i = i + 1) {
        vec.add(data[i]);
      }
      return vec as T;
    }

    if (shape.length == 2) {
      int rows = shape[0];
      int cols = shape[1];
      List<List<double>> mat = [];
      for (int i = 0; i < rows; i = i + 1) {
        List<double> row = [];
        for (int j = 0; j < cols; j = j + 1) {
          row.add(data[(i * cols) + j]);
        }
        mat.add(row);
      }
      return mat as T;
    }

    if (shape.length == 3) {
      int depth  = shape[0];
      int height = shape[1];
      int width  = shape[2];
      List<List<List<double>>> tensor3d = [];
      for (int d = 0; d < depth; d = d + 1) {
        List<List<double>> matrix = [];
        for (int h = 0; h < height; h = h + 1) {
          List<double> row = [];
          for (int w = 0; w < width; w = w + 1) {
            row.add(data[(d * height * width) + (h * width) + w]);
          }
          matrix.add(row);
        }
        tensor3d.add(matrix);
      }
      return tensor3d as T;
    }

    throw Exception('Unflattening beyond 3D is not supported.');
  }

  T get gradValue {
    if (shape.isEmpty) return grad[0] as T;

    if (shape.length == 1) {
      List<double> vec = [];
      for (int i = 0; i < shape[0]; i = i + 1) {
        vec.add(grad[i]);
      }
      return vec as T;
    }

    if (shape.length == 2) {
      int rows = shape[0];
      int cols = shape[1];
      List<List<double>> mat = [];
      for (int i = 0; i < rows; i = i + 1) {
        List<double> row = [];
        for (int j = 0; j < cols; j = j + 1) {
          row.add(grad[(i * cols) + j]);
        }
        mat.add(row);
      }
      return mat as T;
    }

    if (shape.length == 3) {
      int depth  = shape[0];
      int height = shape[1];
      int width  = shape[2];
      List<List<List<double>>> tensor3d = [];
      for (int d = 0; d < depth; d = d + 1) {
        List<List<double>> matrix = [];
        for (int h = 0; h < height; h = h + 1) {
          List<double> row = [];
          for (int w = 0; w < width; w = w + 1) {
            row.add(grad[(d * height * width) + (h * width) + w]);
          }
          matrix.add(row);
        }
        tensor3d.add(matrix);
      }
      return tensor3d as T;
    }

    throw Exception('Unflattening beyond 3D is not supported.');
  }

  // ─────────────────────────────────────────────────────── //
  // Backpropagation & Graph Orchestration
  // ─────────────────────────────────────────────────────── //

  void backward() {
    for (int i = 0; i < grad.length; i = i + 1) {
      grad[i] = 1.0;
    }

    if (creator == null) return;

    List<Node> topo   = [];
    Set<Node> visited = {};

    void buildTopo(Node? node) {
      if (node == null || visited.contains(node)) return;
      visited.add(node);
      for (int i = 0; i < node.inputs.length; i = i + 1) {
        Tensor inputTensor = node.inputs[i];
        if (inputTensor.creator != null) {
          buildTopo(inputTensor.creator);
        }
      }
      topo.add(node);
    }

    buildTopo(creator);

    for (int i = topo.length - 1; i >= 0; i = i - 1) {
      topo[i].backwardFn();
    }
  }

  void zeroGrad() {
    for (int i = 0; i < grad.length; i = i + 1) {
      grad[i] = 0.0;
    }
  }

  bool get isFreed => _freed;
  bool get hasData => !_freed;

  String _getShapeString() {
    if (shape.isEmpty) return '[]';
    if (shape.length == 1) return '[${shape[0]}]';
    if (shape.length == 2) return '[${shape[0]}, ${shape[1]}]';
    if (shape.length == 3) return '[${shape[0]}, ${shape[1]}, ${shape[2]}]';
    return 'unknown';
  }

  // ─────────────────────────────────────────────────────── //
  // Graph Printers
  // ─────────────────────────────────────────────────────── //

  void printGraph() {
    Logger.yellow('Computational Graph [Hybrid CPU/GPU]:', prefix: '📊');
    Set<String> visitedTensors = {};
    _buildGraphString(this, '', true, visitedTensors, true);
  }

  void _buildGraphString(
      Tensor currentTensor,
      String prefix,
      bool isLast,
      Set<String> visited,
      bool isRoot) {

    String tId = currentTensor.id;
    String branchPrefix = prefix;

    if (isLast) {
      branchPrefix = branchPrefix + '└── ';
    } else {
      branchPrefix = branchPrefix + '├── ';
    }

    if (visited.contains(tId)) {
      Logger.red('(Seen again: $tId)', prefix: branchPrefix);
      return;
    }

    visited.add(tId);

    if (currentTensor.creator == null) {
      Logger.green('$tId ${currentTensor._getShapeString()} [CPU] (Leaf: Input)', prefix: branchPrefix);
    } else {
      String opName = currentTensor.creator!.opName;

      if (isRoot) {
        Logger.yellow('$tId ${currentTensor._getShapeString()} [CPU] (Op: $opName)', prefix: branchPrefix);
      } else {
        Logger.blue('$tId ${currentTensor._getShapeString()} [CPU] (Op: $opName)', prefix: branchPrefix);
      }

      Map<String, dynamic>? extras = currentTensor.creator!.extraParams;

      // Check if this node is a portal to the GPU graph
      if (extras.containsKey('gpu_output')) {
        GPUTensor gpuOut = extras['gpu_output'];
        List<GPUTensor> gpuIns = extras['gpu_inputs'];
        List<Tensor> cpuIns = currentTensor.creator!.inputs;

        Map<String, Tensor> boundaryMap = {};
        for (int i = 0; i < gpuIns.length; i = i + 1) {
          boundaryMap[gpuIns[i].id] = cpuIns[i];
        }

        String nextPrefix = prefix;
        if (isLast) {
          nextPrefix = nextPrefix + '    ';
        } else {
          nextPrefix = nextPrefix + '│   ';
        }

        _buildGPUGraphString(gpuOut, nextPrefix, true, visited, boundaryMap);
      } else {
        // Standard CPU traversal
        List<Tensor> inputs = currentTensor.creator!.inputs;
        for (int i = 0; i < inputs.length; i = i + 1) {
          bool isLastChild = false;
          if (i == inputs.length - 1) {
            isLastChild = true;
          }

          String nextPrefix = prefix;
          if (isLast) {
            nextPrefix = nextPrefix + '    ';
          } else {
            nextPrefix = nextPrefix + '│   ';
          }

          _buildGraphString(inputs[i], nextPrefix, isLastChild, visited, false);
        }
      }
    }
  }

  void _buildGPUGraphString(
      GPUTensor currentTensor,
      String prefix,
      bool isLast,
      Set<String> visited,
      Map<String, Tensor> boundaryMap) {

    String tId = currentTensor.id;
    String branchPrefix = prefix;

    if (isLast) {
      branchPrefix = branchPrefix + '└── ';
    } else {
      branchPrefix = branchPrefix + '├── ';
    }

    if (visited.contains(tId)) {
      Logger.red('(Seen again: $tId) [GPU]', prefix: branchPrefix);
      return;
    }

    visited.add(tId);

    String shapeStr = '[';
    for (int i = 0; i < currentTensor.shape.length; i = i + 1) {
      shapeStr = shapeStr + currentTensor.shape[i].toString();
      if (i < currentTensor.shape.length - 1) {
        shapeStr = shapeStr + ', ';
      }
    }
    shapeStr = shapeStr + ']';

    if (currentTensor.creator == null) {
      Logger.green('$tId $shapeStr [GPU] (Leaf: VRAM Input)', prefix: branchPrefix);

      // If this VRAM input maps to a CPU tensor, bridge back to the CPU graph
      if (boundaryMap.containsKey(tId)) {
        Tensor cpuSource = boundaryMap[tId]!;
        String nextPrefix = prefix;
        if (isLast) {
          nextPrefix = nextPrefix + '    ';
        } else {
          nextPrefix = nextPrefix + '│   ';
        }
        _buildGraphString(cpuSource, nextPrefix, true, visited, false);
      }
    } else {
      String opName = currentTensor.creator!.opName;
      Logger.blue('$tId $shapeStr [GPU] (Op: $opName)', prefix: branchPrefix);

      List<GPUTensor> inputs = currentTensor.creator!.inputs;
      for (int i = 0; i < inputs.length; i = i + 1) {
        bool isLastChild = false;
        if (i == inputs.length - 1) {
          isLastChild = true;
        }

        String nextPrefix = prefix;
        if (isLast) {
          nextPrefix = nextPrefix + '    ';
        } else {
          nextPrefix = nextPrefix + '│   ';
        }

        _buildGPUGraphString(inputs[i], nextPrefix, isLastChild, visited, boundaryMap);
      }
    }
  }
}
