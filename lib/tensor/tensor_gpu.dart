import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import '../gpu_version/ffi/OpCodes.dart';
import '../gpu_version/ffi/commandBuffer.dart';
import '../gpu_version/ffi/cudaEngine.dart';
import '../logger.dart';

class GPUNode {
  List<GPUTensor> inputs;
  void Function(CommandBuffer) backwardFn;
  String opName;
  int cost;
  Map<String, dynamic> extraParams;

  GPUNode(
      this.inputs,
      this.backwardFn, {
        this.opName = 'op',
        this.cost = 0,
        this.extraParams = const {},
      });
}

class GPUTensor<T> {
  static int _idCounter = 0;

  String id;
  late List<int> shape;
  GPUNode? creator;

  // Local CPU buffers to mirror VRAM when toCpu() is called
  List<double> data = [];
  List<double> grad = [];

  GPUTensor(dynamic initialValue, {this.creator}) : id = _generateId() {
    // 1. Determine Shape
    if (initialValue is double) {
      shape = [];
    } else if (initialValue is List<double>) {
      shape = [initialValue.length];
    } else if (initialValue is List<List<double>>) {
      int rows = initialValue.length;
      int cols = rows > 0 ? initialValue[0].length : 0;
      shape = [rows, cols];
    } else if (initialValue is List<List<List<double>>>) {
      int depth = initialValue.length;
      int height = depth > 0 ? initialValue[0].length : 0;
      int width = height > 0 ? initialValue[0][0].length : 0;
      shape = [depth, height, width];
    } else {
      throw Exception("Unsupported GPUTensor initialization type: ${initialValue.runtimeType}");
    }

    // 2. Allocate VRAM
    _allocateEmptyInVram();

    // 3. Push Initial Data if provided
    _pushInitialValue(initialValue);
  }

  // Bypasses Dart Lists entirely. Allocates zeros directly in C and pushes to VRAM.
  GPUTensor.empty(List<int> initialShape, {this.creator}) : id = _generateId() {
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }
    _allocateEmptyInVram();
  }
  GPUTensor.randomUniform(List<int> initialShape, double scale, {int? seed, this.creator}) : id = _generateId() {
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }
    _allocateEmptyInVram();

    int finalSeed = seed ?? Random().nextInt(9999999);
    CudaEngine.initRandomUniform(id, scale, finalSeed);
  }




  static String _generateId() {
    int current = _idCounter;
    _idCounter = _idCounter + 1;
    return 't_gpu_$current';
  }

  int _getElementCount() {
    int count = 1;
    for (int i = 0; i < shape.length; i = i + 1) {
      count = count * shape[i];
    }
    return count;
  }

  void _allocateEmptyInVram() {
    int count = _getElementCount();
    Pointer<Float> emptyData = calloc<Float>(count);
    Pointer<Float> emptyGrad = calloc<Float>(count);

    CudaEngine.load(id, emptyData, shape);
    CudaEngine.load(id + '_grad', emptyGrad, shape);

    calloc.free(emptyData);
    calloc.free(emptyGrad);
  }

  void _pushInitialValue(dynamic initialValue) {
    int count = _getElementCount();
    Pointer<Float> ptr = calloc<Float>(count);
    Float32List view = ptr.asTypedList(count);

    if (initialValue is double) {
      view[0] = initialValue;
    } else if (initialValue is List<double>) {
      for (int i = 0; i < initialValue.length; i = i + 1) {
        view[i] = initialValue[i];
      }
    } else if (initialValue is List<List<double>>) {
      int cols = shape[1];
      for (int i = 0; i < initialValue.length; i = i + 1) {
        for (int j = 0; j < initialValue[i].length; j = j + 1) {
          view[(i * cols) + j] = initialValue[i][j];
        }
      }
    } else if (initialValue is List<List<List<double>>>) {
      int height = shape[1];
      int width = shape[2];
      for (int d = 0; d < initialValue.length; d = d + 1) {
        for (int h = 0; h < initialValue[d].length; h = h + 1) {
          for (int w = 0; w < initialValue[d][h].length; w = w + 1) {
            view[(d * height * width) + (h * width) + w] = initialValue[d][h][w];
          }
        }
      }
    }

    CudaEngine.load(id, ptr, shape);
    calloc.free(ptr);
  }

  // ─────────────────────────────────────────────────────── //
  // Data Unflattening (Requires toCpu() to be called first)
  // ─────────────────────────────────────────────────────── //

  T get value {
    if (data.isEmpty) throw Exception("VRAM data not synced to CPU. Call toCpu() first.");
    return _unflatten(data);
  }

  T get gradValue {
    if (grad.isEmpty) throw Exception("VRAM grad not synced to CPU. Call toCpu() first.");
    return _unflatten(grad);
  }

  T _unflatten(List<double> source) {
    if (shape.isEmpty) return source[0] as T;

    if (shape.length == 1) {
      List<double> vec = [];
      for (int i = 0; i < shape[0]; i = i + 1) {
        vec.add(source[i]);
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
          row.add(source[(i * cols) + j]);
        }
        mat.add(row);
      }
      return mat as T;
    }

    if (shape.length == 3) {
      int depth = shape[0];
      int height = shape[1];
      int width = shape[2];
      List<List<List<double>>> tensor3d = [];
      for (int d = 0; d < depth; d = d + 1) {
        List<List<double>> matrix = [];
        for (int h = 0; h < height; h = h + 1) {
          List<double> row = [];
          for (int w = 0; w < width; w = w + 1) {
            row.add(source[(d * height * width) + (h * width) + w]);
          }
          matrix.add(row);
        }
        tensor3d.add(matrix);
      }
      return tensor3d as T;
    }
    throw Exception('Unflattening beyond 3D is not supported.');
  }

  void toCpu() {
    int count = _getElementCount();
    Pointer<Float> pData = calloc<Float>(count);
    Pointer<Float> pGrad = calloc<Float>(count);

    CudaEngine.retrieve(id, pData);
    CudaEngine.retrieve(id + '_grad', pGrad);

    Float32List dataView = pData.asTypedList(count);
    Float32List gradView = pGrad.asTypedList(count);

    data.clear();
    grad.clear();

    for (int i = 0; i < count; i = i + 1) {
      data.add(dataView[i]);
      grad.add(gradView[i]);
    }

    calloc.free(pData);
    calloc.free(pGrad);
  }

  void pushData(List<double> values) {
    int count = values.length;
    Pointer<Float> ptr = calloc<Float>(count);
    Float32List view = ptr.asTypedList(count);

    for (int i = 0; i < count; i = i + 1) {
      view[i] = values[i];
    }

    CudaEngine.load(id, ptr, shape);
    calloc.free(ptr);
  }


  void backward(CommandBuffer backwardTape, {bool fillOnes = true}) {
    if (fillOnes == true) {
      backwardTape.putInt(OP_FILL);
      backwardTape.putString(id + '_grad');
      backwardTape.putFloat(1.0);
    }

    if (creator == null) return;

    List<GPUNode> topo = [];
    Set<GPUNode> visited = {};

    void buildTopo(GPUNode? node) {
      if (node == null || visited.contains(node)) return;
      visited.add(node);
      for (int i = 0; i < node.inputs.length; i = i + 1) {
        GPUTensor inputTensor = node.inputs[i];
        if (inputTensor.creator != null) {
          buildTopo(inputTensor.creator);
        }
      }
      topo.add(node);
    }

    buildTopo(creator);

    for (int i = topo.length - 1; i >= 0; i = i - 1) {
      topo[i].backwardFn(backwardTape);
    }
  }

  void free() {
    CudaEngine.free(id);
    CudaEngine.free(id + '_grad');
  }

  // ─────────────────────────────────────────────────────── //
  // Graph Printing
  // ─────────────────────────────────────────────────────── //

  void printGraph() {
    Logger.yellow('GPU Computational Graph:', prefix: '🚀');
    Set<String> visited = {};
    _buildGPUGraphString(this, '', true, visited);
  }

  void _buildGPUGraphString(GPUTensor current, String prefix, bool isLast, Set<String> visited) {
    String branchPrefix = isLast ? prefix + '└── ' : prefix + '├── ';

    if (visited.contains(current.id)) {
      Logger.red('(Seen again: ${current.id}) [GPU]', prefix: branchPrefix);
      return;
    }
    visited.add(current.id);

    String shapeStr = '[${current.shape.join(", ")}]';

    if (current.creator == null) {
      Logger.green('${current.id} $shapeStr [GPU] (Leaf: VRAM Input)', prefix: branchPrefix);
    } else {
      Logger.blue('${current.id} $shapeStr [GPU] (Op: ${current.creator!.opName})', prefix: branchPrefix);

      List<GPUTensor> inputs = current.creator!.inputs;
      for (int i = 0; i < inputs.length; i = i + 1) {
        String nextPrefix = isLast ? prefix + '    ' : prefix + '│   ';
        _buildGPUGraphString(inputs[i], nextPrefix, i == inputs.length - 1, visited);
      }
    }
  }

  List<GPUTensor> getAllTensorsInGraph(GPUTensor root) {
    List<GPUTensor> all = <GPUTensor>[];
    Set<String> visited = <String>{};

    void traverse(GPUTensor node) {
      if (visited.contains(node.id)) {
        return;
      }
      visited.add(node.id);
      all.add(node);

      if (node.creator != null) {
        for (int i = 0; i < node.creator!.inputs.length; i = i + 1) {
          traverse(node.creator!.inputs[i]);
        }
      }
    }

    traverse(root);
    return all;
  }

  void zeroGrad(CommandBuffer tape) {
    tape.putInt(OP_ZERO_GRAD);
    tape.putString(id + '_grad');
  }

  void zeroGraphGrads(CommandBuffer tape) {
    List<GPUTensor> allTensors = getAllTensorsInGraph(this);
    for (int i = 0; i < allTensors.length; i = i + 1) {
      allTensors[i].zeroGrad(tape);
    }
  }
}