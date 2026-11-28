import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '/full_library.dart';

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
/// Each GPUTensor is both holding gradient value and its actual data value. Initialisation is to be done via its constructor.
///
/// IMPORTANT: The datatype [T] is not predefined as a standard value. To ensure compatibility with the framework the use of [Scalar],[Vector],[Matrix] and [Tensor3D] is recommended.
///
/// GPUTensors are NOT allocated on construction. Every constructor records its allocation into the provided [AllocBuffer], which must be executed via [AllocBuffer.execute] before running tapes.
/// Accessing GPUTensor values can be done via its local Buffers [data] and [grad] after [toCpu] was called to update the buffers. Forgetting to update will result in outdated or empty buffers.
class GPUTensor<T> {
  static int _idCounter = 0;

  /// Global default for saying if allocated GPUTensors should hold gradient by default.
  static bool globalGrad = true;

  /// Global identifier to find and query tensors. [id] is set to "t_gpu_[_idCounter] by default. It is recommended NOT to set this value manually outside of creation.
  /// Uniqueness is to be handled manually when setting the name.
  late String id;
  String get gradId => '${id}_grad';

  /// Stores as which shape the GPUTensors value should be interpreted as. 0-3 dimensions are supported currently (2026-09).
  late List<int> shape;

  /// [GPUNode] [creator] stores the GPUNode that is responsible for the creation of this GPUTensor. This allows for backward gradient propagation as well as tracing the origin path of this GPUTensor.
  GPUNode? creator;

  /// Unoptimised List of the retrieved last value of the GPUTensor.
  List<double> data = <double>[];

  /// Unoptimised List of the retrieved last gradient of the GPUTensor.
  List<double> grad = <double>[];

  /// Map to capture the sub GPUTensors that may have been allocate during creation.
  Map<String, GPUTensor> subMap = <String, GPUTensor>{};

  late bool localGrad;

  GPUTensor(
    T initialValue,
    AllocBuffer allocTape, {
    this.creator,
    String? id,
    bool? requiresGrad,
    String? sourceOp,
  }) {
    localGrad = requiresGrad ?? globalGrad;
    this.id = id ?? _generateId();

    if (initialValue is double) {
      shape = <int>[];
    } else if (initialValue is List<double>) {
      shape = <int>[initialValue.length];
    } else if (initialValue is List<List<double>>) {
      int rows = initialValue.length;
      int cols = rows > 0 ? initialValue[0].length : 0;
      shape = <int>[rows, cols];
    } else if (initialValue is List<List<List<double>>>) {
      int depth = initialValue.length;
      int height = depth > 0 ? initialValue[0].length : 0;
      int width = height > 0 ? initialValue[0][0].length : 0;
      shape = <int>[depth, height, width];
    } else {
      throw ArgumentError("Invalid initialization for GPUTensor<$T>. "
          "Expected Scalar, Vector, Matrix, or Tensor3D, got: ${initialValue.runtimeType}");
    }

    String opTag = sourceOp ?? creator?.opName ?? 'tensor';
    allocTape.putData(this.id, shape, _flattenToFloat32(initialValue),
        sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  GPUTensor.scalar(
    Scalar? initialValue,
    AllocBuffer allocTape, {
    this.creator,
    this.id = "EMPTY",
    bool? requiresGrad,
    String? sourceOp,
  }) {
    localGrad = requiresGrad ?? globalGrad;
    if (id == "EMPTY") {
      id = _generateId();
    }
    shape = <int>[];

    String opTag = sourceOp ?? creator?.opName ?? 'scalar';
    if (initialValue != null) {
      allocTape.putData(id, shape, _flattenToFloat32(initialValue),
          sourceOp: opTag);
    } else {
      allocTape.putEmpty(id, shape, sourceOp: opTag);
    }
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  GPUTensor.vector(
    Vector initialValue,
    AllocBuffer allocTape, {
    this.creator,
    this.id = "EMPTY",
    bool? requiresGrad,
    String? sourceOp,
  }) {
    localGrad = requiresGrad ?? globalGrad;
    if (id == "EMPTY") {
      id = _generateId();
    }
    shape = <int>[initialValue.length];

    String opTag = sourceOp ?? creator?.opName ?? 'vector';
    allocTape.putData(id, shape, _flattenToFloat32(initialValue),
        sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  GPUTensor.matrix(
    Matrix initialValue,
    AllocBuffer allocTape, {
    this.creator,
    this.id = "EMPTY",
    bool? requiresGrad,
    String? sourceOp,
  }) {
    localGrad = requiresGrad ?? globalGrad;
    if (id == "EMPTY") {
      id = _generateId();
    }
    int rows = initialValue.length;
    int cols = rows > 0 ? initialValue[0].length : 0;
    shape = <int>[rows, cols];

    String opTag = sourceOp ?? creator?.opName ?? 'matrix';
    allocTape.putData(id, shape, _flattenToFloat32(initialValue),
        sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  GPUTensor.tensor3D(
    Tensor3D initialValue,
    AllocBuffer allocTape, {
    this.creator,
    this.id = "EMPTY",
    bool? requiresGrad,
    String? sourceOp,
  }) {
    localGrad = requiresGrad ?? globalGrad;
    if (id == "EMPTY") {
      id = _generateId();
    }
    int depth = initialValue.length;
    int height = depth > 0 ? initialValue[0].length : 0;
    int width = height > 0 ? initialValue[0][0].length : 0;
    shape = <int>[depth, height, width];

    String opTag = sourceOp ?? creator?.opName ?? 'tensor3D';
    allocTape.putData(id, shape, _flattenToFloat32(initialValue),
        sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  /// Create a GPUTensor with the given shape and records an empty allocation in [allocTape].
  /// Additionally the [creator] GPUNode can be set.
  GPUTensor.empty(
    List<int> initialShape,
    AllocBuffer allocTape, {
    this.creator,
    bool? requiresGrad,
    String? sourceOp,
  }) : id = _generateId() {
    localGrad = requiresGrad ?? globalGrad;
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }

    String opTag = sourceOp ?? creator?.opName ?? 'empty';
    allocTape.putEmpty(id, shape, sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  /// Create a GPUTensor with the given shape and records a random uniform initialization in [allocTape].
  /// Additionally the [creator] GPUNode can be set as well as a seed for randomization.
  GPUTensor.randomUniform(
    List<int> initialShape,
    double scale,
    AllocBuffer allocTape, {
    int? seed,
    this.creator,
    bool? requiresGrad,
    String? sourceOp,
  }) : id = _generateId() {
    localGrad = requiresGrad ?? globalGrad;
    shape = <int>[];
    for (int i = 0; i < initialShape.length; i = i + 1) {
      shape.add(initialShape[i]);
    }

    int finalSeed = seed ?? Random().nextInt(9999999);
    String opTag = sourceOp ?? creator?.opName ?? 'randomUniform';

    allocTape.putEmpty(id, shape, sourceOp: opTag);
    allocTape.putRandomUniform(id, shape, scale, finalSeed, sourceOp: opTag);
    if (localGrad == true) {
      allocTape.putEmpty(gradId, shape, sourceOp: '$opTag:grad', isGrad: true);
    }
  }

  /// Converts [data] via [shape] and returns the value as [T].
  T get value {
    if (data.isEmpty) {
      throw Exception("VRAM data not synced to CPU. Call toCpu() first.");
    }
    return _unflatten(data);
  }

  /// Converts [grad] via [shape] and returns the value as [T].
  T get gradValue {
    if (grad.isEmpty) {
      throw Exception("VRAM grad not synced to CPU. Call toCpu() first.");
    }
    return _unflatten(grad);
  }

  void dataToCpu() {
    int count = _getElementCount();

    using((Arena arena) {
      Pointer<Float> pData = arena<Float>(count);
      GPUEngine.retrieve(id, pData);
      Float32List dataView = pData.asTypedList(count);

      data.clear();
      data.addAll(dataView);
    });
  }

  void gradToCpu() {
    if (localGrad == false) return;

    int count = _getElementCount();

    using((Arena arena) {
      Pointer<Float> pGrad = arena<Float>(count);
      GPUEngine.retrieve(gradId, pGrad);
      Float32List gradView = pGrad.asTypedList(count);

      grad.clear();
      grad.addAll(gradView);
    });
  }

  /// Retrieves the current state of the GPUTensor from VRAM and writes it to the [data] and [grad] buffers.
  void toCpu() {
    dataToCpu();
    gradToCpu();
  }

  /// Manually pushes values to overwrite the GPUTensors values in VRAM.
  void pushData(List<double> values) {
    int count = values.length;

    using((Arena arena) {
      Pointer<Float> ptr = arena<Float>(count);
      Float32List view = ptr.asTypedList(count);

      view.setAll(0, values);

      GPUEngine.load(id, ptr, shape);
    });
  }

  /// Starting point of gradient calculation. The network of GPUNodes and GPUTensors traverses in topological order over all GPUTensors writing their given commands to calculate the gradients to [backwardTape].
  /// To actually calculate the values this tape is required to be executed via [GPUEngine.run(backwardTape)].
  /// If [fillOnes] is set to true, it pushes a command to fill the gradient of this tensor with 1.0.
  void backward(CommandBuffer backwardTape, {bool fillOnes = true}) {
    if (localGrad == false) return;

    if (fillOnes == true) {
      backwardTape.putInt(OP_FILL);
      backwardTape.putString(gradId);
      backwardTape.putFloat(1.0);
    }

    if (creator == null) return;

    List<GPUNode> topo = <GPUNode>[];
    Set<GPUNode> visited = <GPUNode>{};

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
    for (GPUTensor tensor in subMap.values) {
      tensor.free();
    }

    GPUEngine.free(id);

    if (localGrad == true) {
      GPUEngine.free('${id}_grad');
    }
  }

  /// Allows the exact printing of the current compute graph for this GPUTensor. Topological Depth is indicated via indents.
  void printGraph() {
    Logger.yellow('GPU Computational Graph:', prefix: '🚀');
    Set<String> visited = <String>{};
    _buildGPUGraphString(this, '', true, visited);
  }

  /// Writes the command to set the gradient of this GPUTensor to 0.
  void zeroGrad(CommandBuffer tape) {
    if (localGrad == false) return;
    tape.putInt(OP_ZERO_GRAD);
    tape.putString(gradId);
  }

  T _unflatten(List<double> source) {
    if (shape.isEmpty) return source[0] as T;

    if (shape.length == 1) {
      List<double> vec = <double>[];
      for (int i = 0; i < shape[0]; i = i + 1) {
        vec.add(source[i]);
      }
      return vec as T;
    }

    if (shape.length == 2) {
      int rows = shape[0];
      int cols = shape[1];
      List<List<double>> mat = <List<double>>[];
      for (int i = 0; i < rows; i = i + 1) {
        List<double> row = <double>[];
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
      List<List<List<double>>> tensor3d = <List<List<double>>>[];
      for (int d = 0; d < depth; d = d + 1) {
        List<List<double>> matrix = <List<double>>[];
        for (int h = 0; h < height; h = h + 1) {
          List<double> row = <double>[];
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

  void _buildGPUGraphString(
      GPUTensor current, String prefix, bool isLast, Set<String> visited) {
    String branchPrefix = isLast ? '$prefix└── ' : '$prefix├── ';

    if (visited.contains(current.id)) {
      Logger.red('(Seen again: ${current.id}) [GPU]', prefix: branchPrefix);
      return;
    }
    visited.add(current.id);

    String shapeStr = '[${current.shape.join(", ")}]';

    if (current.creator == null) {
      Logger.green('${current.id} $shapeStr [GPU] (Leaf: VRAM Input)',
          prefix: branchPrefix);
    } else {
      Logger.blue(
          '${current.id} $shapeStr [GPU] (Op: ${current.creator!.opName})',
          prefix: branchPrefix);

      List<GPUTensor> inputs = current.creator!.inputs;
      for (int i = 0; i < inputs.length; i = i + 1) {
        String nextPrefix = isLast ? '$prefix    ' : '$prefix│   ';
        _buildGPUGraphString(
            inputs[i], nextPrefix, i == inputs.length - 1, visited);
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

  /// Helper method to flatten data to be stored in the buffer itself.
  Float32List _flattenToFloat32(dynamic initialValue) {
    int count = _getElementCount();
    Float32List flat = Float32List(count);

    if (initialValue is double) {
      flat[0] = initialValue;
    } else if (initialValue is List<double>) {
      for (int i = 0; i < initialValue.length; i = i + 1) {
        flat[i] = initialValue[i];
      }
    } else if (initialValue is List<List<double>>) {
      int cols = shape[1];
      for (int i = 0; i < initialValue.length; i = i + 1) {
        for (int j = 0; j < initialValue[i].length; j = j + 1) {
          flat[(i * cols) + j] = initialValue[i][j];
        }
      }
    } else if (initialValue is List<List<List<double>>>) {
      int height = shape[1];
      int width = shape[2];
      for (int d = 0; d < initialValue.length; d = d + 1) {
        for (int h = 0; h < initialValue[d].length; h = h + 1) {
          for (int w = 0; w < initialValue[d][h].length; w = w + 1) {
            flat[(d * height * width) + (h * width) + w] =
                initialValue[d][h][w];
          }
        }
      }
    }

    return flat;
  }
}
