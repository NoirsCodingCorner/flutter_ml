import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter_ml/full_library.dart';
import '../logger.dart';



class GPUNode {
  List<GPUTensor> inputs;
  void Function(CommandBuffer) backwardFn;
  String opName;
  int cost;
  Map<String, dynamic> extraParams;
  /// A GPUNode connects different [GPUTensor] together. In contrast to [Node] GPUNodes are used to generate a [CommandBuffer] creating a runnable list of mathematical functions.
  /// GPUNodes create a directed acyclical graph structure and each GPUNode can be viewed as functions holding reference to its input GPUTensors.
  ///
  /// `opName` is used to give a function a recognisable name for debugging and traceability.
  ///
  /// `extraParams` allows to store additional arguments fe. an integer describing recursive depth used in the function.
  ///
  /// The additional parameter for the backwards function is for the writing of the operation to the backwards `CommandBuffer`.
  /// GPUNodes are mostly used within the creation and coupling of [GPUTensor] and may not be required unless explicitly defining new functions.
  ///
  /// Example:
  /// ```dart
  /// out.creator = GPUNode(
  ///     <GPUTensor>[a, b],
  ///         (CommandBuffer bTape) {
  ///       bTape.putInt(OP_ADD_INTO);
  ///       bTape.putString('${out.id}_grad');
  ///       bTape.putString('${a.id}_grad');
  ///
  ///       bTape.putInt(OP_ADD_INTO);
  ///       bTape.putString('${out.id}_grad');
  ///       bTape.putString('${b.id}_grad');
  ///     },
  ///     opName: 'addGPU',
  ///   );
  /// ```
  GPUNode(
      this.inputs,
      this.backwardFn, {
        this.opName = 'op',
        this.cost = 0,
        this.extraParams = const {},
      });
}

/// A [GPUTensor] in this library is used as the fundamental unit to hold structured collections of `float` values on the GPU. For CPU computation please refer to [Tensor].
/// These values are stored on the GPU (usually inside the VRAM) with a unique identifier `id`.
/// Globally a counter called `idCounter` ensures uniqueness of Tensors up to the integer limit of instances.
///
/// Each GPUTensor is boath holding gradient value and its actual data value. Initialisation is to be done via its constructor.
///
/// IMPORTANT: The datatype [T] is not predefined as a standard value. To ensure compatibility with the framework the use of [Scalar],[Vector],[Matrix] and [Tensor3D] is recommended.
///
/// GPUTensors are NOT managed automatically. Allocating is done on construction. To free the tensor again use the [.free] method.
/// Accessing GPUTensor values can be done via its local Buffers [data] and [grad] after [toCpu] was called to update the buffers. Forgetting to update will result in outdated or empty buffers.
class GPUTensor<T> {
  static int _idCounter = 0;

  /// Global identifier to find and query tensors. [id] is set to "t_gpu_[_idCounter] by default. It is recommended NOT to set this value manually outside of creation.
  /// Uniqueness is to be handled manually when setting the name.
  late String id;

  /// Stores as which shape the GPUTensors value should be interpreted as. 0-3 dimensions are supported currently (2026-09).
  late List<int> shape;

  /// [GPUNode] [creator] stores the GPUNode that is responsible for the creation of this GPUTensor. This allows for backward gradient propagation as well as tracing the origin path of this GPUTensor.
  GPUNode? creator;

  /// Unoptimised List of the retrieved last value of the GPUTensor.
  List<double> data = [];
  /// Unoptimised List of the retrieved last gradient of the GPUTensor.
  List<double> grad = [];

  /// Map to capture the sub GPUTensors that may have been allocate during creation.
  Map<String,GPUTensor>subMap={};


  /// Creates a [GPUTensor] object storing and managing the value [initialValue] given. Supported types for GPUTensors are:
  /// [Scalar],[Vector],[Matrix] and [Tensor3D]. Short versions of those types are [Sc],[Vec],[Mat] and [T3D].
  /// GPUTensors can be created the following ways:
  /// ```dart
  /// GPUTensor<Scalar>scalar=GPUTensor<Scalar>(3.0);
  /// GPUTensor<Vector>vector=GPUTensor<Vector>([1.0,2.0,3.0]);
  /// GPUTensor<Matrix>matrix=GPUTensor<Matrix>([[1.0,2.0],[3.0,4.0],);
  /// ```
  ///
  /// Additionally for faster allocation, the following methods can be used:
  /// ```dart
  /// GPUTensor<T>a=GPUTensor.empty([1]);
  /// GPUTensor<T>a=GPUTensor.randomUniform([1,2,3],1.0);
  /// ```
  ///
  /// Regardless of creation method, a [GPUNode] [creator] can be given. This parameter will be used for calculation of connected input GPUTensors gradient values.
  /// If no [creator] is given, this GPUTensor can only be used as an input inside of the computational structure.
  ///
  /// Connecting GPUTensors is one of the main focus points of this engine to enable automatic gradient calculation and can be done like the following example:
  /// ```dart
  /// GPUTensor<T> addGPU<T>(GPUTensor<T> a, GPUTensor<T> b, CommandBuffer tape) {
  ///   GPUTensor<T> out = GPUTensor<T>.empty(a.shape);
  ///
  ///   tape.putInt(OP_ADD);
  ///   tape.putString(a.id);
  ///   tape.putString(b.id);
  ///   tape.putString(out.id);
  ///
  ///   out.creator = GPUNode(
  ///     <GPUTensor>[a, b],
  ///         (CommandBuffer bTape) {
  ///       bTape.putInt(OP_ADD_INTO);
  ///       bTape.putString('${out.id}_grad');
  ///       bTape.putString('${a.id}_grad');
  ///
  ///       bTape.putInt(OP_ADD_INTO);
  ///       bTape.putString('${out.id}_grad');
  ///       bTape.putString('${b.id}_grad');
  ///     },
  ///     opName: 'addGPU',
  ///   );
  ///   return out;
  /// }
  /// ```
  /// The raw [data] can be accessed via [.data]. To return a structured value [.value] converts it via [shape] into [T].
  /// The raw [grad] can be accessed via [.grad]. To return a structured value [.gradValue] converts it via [shape] into [T].
  /// Both require [toCpu] to be called directly before that if the value is to be pulled from VRAM.
  ///
  GPUTensor(dynamic initialValue, {this.creator, this.id="EMPTY"}) {
    if(id=="EMPTY"){
      id = _generateId();
    }
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

  /// Create a GPUTensor with the given shape and fills the GPUTensor with 0.0.
  /// Additionally the [creator] GPUNode can be set.
  GPUTensor.empty(List<int> initialShape, {this.creator}) : id = _generateId() {
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }
    _allocateEmptyInVram();
  }

  /// Create a GPUTensor with the given shape and fills the GPUTensor with random initialised values in range of provided scale parameter.
  /// Additionally the [creator] GPUNode can be set as well as a seed for randomization.
  GPUTensor.randomUniform(List<int> initialShape, double scale, {int? seed, this.creator}) : id = _generateId() {
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }
    _allocateEmptyInVram();

    int finalSeed = seed ?? Random().nextInt(9999999);
    GPUEngine.initRandomUniform(id, scale, finalSeed);
  }

  /// Converts [data] via [shape] and returns the value as [T].
  T get value {
    if (data.isEmpty) throw Exception("VRAM data not synced to CPU. Call toCpu() first.");
    return _unflatten(data);
  }
  /// Converts [grad] via [shape] and returns the value as [T].
  T get gradValue {
    if (grad.isEmpty) throw Exception("VRAM grad not synced to CPU. Call toCpu() first.");
    return _unflatten(grad);
  }


  /// Retrieves the current state of the GPUTensor from VRAM and writes it to the [data] and [grad] buffers.
  void toCpu() {
    int count = _getElementCount();
    Pointer<Float> pData = calloc<Float>(count);
    Pointer<Float> pGrad = calloc<Float>(count);

    GPUEngine.retrieve(id, pData);
    GPUEngine.retrieve('${id}_grad', pGrad);

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

  /// Manually pushes values to overwrite the GPUTensors values in VRAM.
  void pushData(List<double> values) {
    int count = values.length;
    Pointer<Float> ptr = calloc<Float>(count);
    Float32List view = ptr.asTypedList(count);

    for (int i = 0; i < count; i = i + 1) {
      view[i] = values[i];
    }

    GPUEngine.load(id, ptr, shape);
    calloc.free(ptr);
  }

  /// Staring point of gradient calculation. The network of GPUNodes and GPUTensors traverses in topological oder over all GPUTensors writing their given commands to calculate the gradients to [backwardsTape].
  /// To actually calculate the values this tape is required to be executed via [GPUEngine.run(backwardTape)].
  /// If [fillOnes] is set to true, it pushes a command to fill the gradient of this tensor with 1.0.
  void backward(CommandBuffer backwardTape, {bool fillOnes = true}) {
    if (fillOnes == true) {
      backwardTape.putInt(OP_FILL);
      backwardTape.putString('${id}_grad');
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

  /// Frees the allocated GPU Memory for both value and gradient.
  void free() {
    for(GPUTensor tensor in subMap.values){
      tensor.free();
    }

    GPUEngine.free(id);
    GPUEngine.free('${id}_grad');
  }

  /// Allows the exact printing of the current compute graph for this GPUTensor. Topological Depth is indicated via indents.
  /// Colors indicate tensor types. Green is indication of an input Tensor, blue an intermediate Tensor and Yellow the resulting output of the Network.
  /// Tensor name and shape are listed as `t_tensorid [1,4,5,...]`
  /// Example for a simple neuronal network:
  /// ```dart
  /// --- Inspecting Computational Graph ---
  /// 🚀 GPU Computational Graph:
  /// └──  t_gpu_4 [3] [GPU] (Op: elementWiseMultiplyGPU)
  ///     ├──  t_gpu_2 [3] [GPU] (Leaf: VRAM Input)
  ///     └──  t_gpu_3 [3] [GPU] (Op: addVectorGPU)
  ///         ├──  t_gpu_0 [3] [GPU] (Leaf: VRAM Input)
  ///         └──  t_gpu_1 [3] [GPU] (Leaf: VRAM Input)
  /// ```
  /// For decoding od the CommandBuffer to inspect what gets calculated in what order on the gpu a TapeDecoder can be used:
  /// ```dart
  /// TapeDecoder(buffer.bytes()).decode();
  /// ->
  /// 📜 --- Decoding Execution Tape (62 bytes) ---
  /// ℹ️ OP_ADD: t_gpu_3 = t_gpu_0 op t_gpu_1
  /// ℹ️ OP_MULTIPLY: t_gpu_4 = t_gpu_2 op t_gpu_3
  /// 📜 --- End of Tape ---
  ///
  /// ```
  void printGraph() {
    Logger.yellow('GPU Computational Graph:', prefix: '🚀');
    Set<String> visited = {};
    _buildGPUGraphString(this, '', true, visited);
  }

  /// Traverses the graph and returns a list of all GPUTensors this root GPUTensor is build from.
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

  /// Writes the command to set the gradient of this GPUTensor to 0.
  void zeroGrad(CommandBuffer tape) {
    tape.putInt(OP_ZERO_GRAD);
    tape.putString('${id}_grad');
  }

  /// Finds every GPUTensor the current GPUTensor depends on and sets their gradients to 0.
  void zeroGraphGrads(CommandBuffer tape) {
    List<GPUTensor> allTensors = getAllTensorsInGraph(this);
    for (int i = 0; i < allTensors.length; i = i + 1) {
      allTensors[i].zeroGrad(tape);
    }
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

  void _buildGPUGraphString(GPUTensor current, String prefix, bool isLast, Set<String> visited) {
    String branchPrefix = isLast ? '$prefix└── ' : '$prefix├── ';

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
        String nextPrefix = isLast ? '$prefix    ' : '$prefix│   ';
        _buildGPUGraphString(inputs[i], nextPrefix, i == inputs.length - 1, visited);
      }
    }
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

    GPUEngine.load(id, emptyData, shape);
    GPUEngine.load('${id}_grad', emptyGrad, shape);

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

    GPUEngine.load(id, ptr, shape);
    calloc.free(ptr);
  }

}