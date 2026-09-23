import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';
import '../tensor/tensor_gpu.dart';
import 'dart:typed_data';
import '../logger.dart';



class Node {

  List<Tensor> inputs;
  Function backwardFn;
  String opName;
  int cost;
  Map<String, dynamic> extraParams;

  /// A Node connects different [Tensor] together. Nodes can be viewed inside a computational graph as functions holding references to its input tensors.
  /// To allow backpropagation through the network `backwardFn` is stored as a reference to the given method.
  /// Nodes are used for eager cpu execution of functions.
  ///
  /// `cost` is used as an approximate value to allow diagnosing and debugging a systems performance.
  ///
  /// `opName` is used to give a function a recognisable name for debugging and traceability.
  ///
  /// `extraParams` allows to store additional arguments fe. an integer describing recursive depth used in the function.
  ///
  /// Nodes are mostly used within the creation and coupling of new [Tensor],
  ///
  /// Example:
  /// ```dart
  ///   out.creator = Node(
  ///     [a, b],
  ///         () {
  ///       a.grad[0] = a.grad[0] + out.grad[0];
  ///       b.grad[0] = b.grad[0] + out.grad[0];
  ///     },
  ///     opName: 'add',
  ///     cost: 1,
  ///   );
  /// ```
  Node(
      this.inputs,
      this.backwardFn, {
        this.opName = 'op',
        this.cost = 0,
        this.extraParams = const {},
      });
}


/// A [Tensor] in this library is used as the fundamental unit to hold a structured collection of `float` values on the CPU.
/// These values are stored for performance enhancement inside of [Float32List].
/// Globally a counter called `idCounter` ensures uniqueness of Tensors up to the integer limit of instances.
///
/// Each Tensor is both holding gradient value and its actual data value. Initialisation is to be done via its constructor.
///
/// IMPORTANT: The datatype [T] is not predefined as a standard value. To ensure compatibility with the framework the use of [Scalar],[Vector],[Matrix] and [Tensor3D] is recommended.
///
/// Tensors that end up orphaned are deleted using the [Finalizer] class but for maximum performance manual management
/// is recommended. Boolean [_freed] is used as an additional safety feature allowing to see if the memory allocated has already been freed.
class Tensor<T> {
  static int _idCounter = 0;

  static final Finalizer<(Pointer<Float>, Pointer<Float>)> _finalizer = Finalizer((ptrs) {
    calloc.free(ptrs.$1);
    calloc.free(ptrs.$2);
  });

  /// Global identifier to find and query tensors. [id] is set to "t_[_idCounter] by default. Id can be set anytime on CPU-bound Tensors.
  late String id;


  /// C pointer to the memory location of [data].
  late Pointer<Float> dataPtr;
  /// `Rolled out List of the Tensors value.
  late Float32List data;
  /// C pointer to the memory location of [grad].
  late Pointer<Float> gradPtr;
  /// `Rolled out List of the Tensors gradient value.
  late Float32List grad;

  /// Stores as which shape the Tensors value should be interpreted as. 0-3 dimensions are supported currently (2026-09).
  late List<int> shape;

  /// [Node] [creator] stores the node that is responsible for the creation of this Tensor. This allows for backward gradient propagation as well as tracing the origin path of this Tensor.
  Node? creator;

  /// Stores whether the allocated storage used for this Tensor has been freed. `true` means the Tensors value is no longer valid.
  bool _freed = false;

  // ─────────────────────────────────────────────────────── //
  // Initialization
  // ─────────────────────────────────────────────────────── //

  /// Creates a [Tensor] object storing and managing the value [initialValue] given. Supported types for Tensor initialisation are:
  /// [Scalar],[Vector],[Matrix] and [Tensor3D]. Short versions of those types are [Sc],[Vec],[Mat] and [T3D].
  /// Tensors can be created the following ways:
  /// ```dart
  /// Tensor<Scalar>scalar=Tensor<Scalar>(3.0);
  /// Tensor<Vector>vector=Tensor<Vector>([1.0,2.0,3.0]);
  /// Tensor<Matrix>matrix=Tensor<Matrix>([[1.0,2.0],[3.0,4.0],);
  /// ```
  ///
  /// Additionally a [Node] [creator] can be given. This parameter will be used for calculation of connected input Tensors gradient values.
  /// If no [creator] is given this Tensor can only be used as an input inside of a computational structure.
  ///
  /// Connecting Tensors is one of the main focus points of this engine and can be done like the following example:
  ///
  /// ```dart
  /// Tensor<Scalar> add(Tensor<Scalar> a, Tensor<Scalar> b) {
  ///   Tensor<Scalar> out = Tensor<Scalar>(a.data[0] + b.data[0]);
  ///
  ///   out.creator = Node(
  ///     [a, b],
  ///         () {
  ///       a.grad[0] = a.grad[0] + out.grad[0];
  ///       b.grad[0] = b.grad[0] + out.grad[0];
  ///     },
  ///     opName: 'add',
  ///     cost: 1,
  ///   );
  ///   return out;
  /// }
  /// ```
  /// The raw [data] can be accessed via [.data]. To return a structured value [.value] converts it via [shape] into [T].
  /// The raw [grad] can be accessed via [.grad]. To return a structured value [.gradValue] converts it via [shape] into [T].
  ///
  /// A Tensor carries multiple variables:
  ///
  /// [id] : A String id to allow easy identification. Can be set manually for variable tracing in development.
  ///
  /// [dataPtr] : A direct System pointer on the native C memory where the [data] is located.
  ///
  /// [gradPtr] : A direct System pointer on the native C memory where the [grad] is located.
  ///
  /// [shape] : `List<int>`of the dimensions of the stored arrays. Shapes of [grad] and [data] are always the same.
  ///
  /// [creator] : An optional [Node] parameter to tie the tensor as a result of a specific operation.
  Tensor(dynamic initialValue, {this.creator}) {
    id = 't_$_idCounter';
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

    _finalizer.attach(this, (dataPtr, gradPtr), detach: this);

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


  /// Helper function to return the total element amount of the Tensor.
  static int _elementCount(List<int> shape) {
    if (shape.isEmpty) return 1;
    int n = 1;
    for (int i = 0; i < shape.length; i = i + 1) {
      n = n * shape[i];
    }
    return n;
  }

  /// Frees the allocated C memory and sets [_freed] to `true`.
  void free() {
    if (_freed) return;

    calloc.free(dataPtr);
    calloc.free(gradPtr);

    _finalizer.detach(this);
    _freed = true;
  }

  /// Converts [data] via [shape] and returns the value as [T].
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
  /// Converts [grad] via [shape] and returns the value as [T].
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

  /// Starting point of gradient calculation. The network of Nodes and Tensors traverses in topological order over all Tensors affecting this Tensor adding a given gradient, calculated via their [creator].
  /// The value of [startingGrad] affects the first gradient given into the backwards process and is set to `1.0` by default.
  void backward({double startingGrad=1.0}) {
    for (int i = 0; i < grad.length; i = i + 1) {
      grad[i] = startingGrad;
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

  /// Traverses the computational Graph which created this Tensor, setting every value of every gradient to `0.0`. This method in machine learning is to be used in between
  /// training iterations. Forgetting this method may result in oscillating or exploding gradient values.
  void zeroGrad() {
    for (int i = 0; i < grad.length; i = i + 1) {
      grad[i] = 0.0;
    }
  }

  bool get isFreed => _freed;
  bool get hasData => !_freed;

  /// Prints out the Shape of the network as a String.
  /// Example:
  /// ```dart
  /// "[1,2,3]"
  /// ```
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

  /// Allows the exact printing of the current compute graph for this Tensor. Topological Depth is indicated via indents.
  /// Colors indicate tensor types. Green is indication of an input Tensor, blue an intermediate Tensor and Yellow the resulting output of the Network.
  /// Tensor name and shape are listed as `t_tensorid [1,4,5,...]`
  /// Example for a simple neuronal network:
  /// ```dart
  /// --- Inspecting Computational Graph ---
  /// 📊 Computational Graph [Hybrid CPU/GPU]:
  /// └──  t_180019 [] [CPU] (Op: mse_vector)
  ///     ├──  t_180018 [1] [CPU] (Op: sigmoid_vector)
  ///     │   └──  t_180017 [1] [CPU] (Op: add_vector)
  ///     │       ├──  t_180016 [1] [CPU] (Op: matVecMul)
  ///     │       │   ├──  t_6 [1, 8] [CPU] (Leaf: Input)
  ///     │       │   └──  t_180015 [8] [CPU] (Op: relu_vector)
  ///     │       │       └──  t_180014 [8] [CPU] (Op: add_vector)
  ///     │       │           ├──  t_180013 [8] [CPU] (Op: matVecMul)
  ///     │       │           │   ├──  t_1 [8, 2] [CPU] (Leaf: Input)
  ///     │       │           │   └──  t_180011 [2] [CPU] (Leaf: Input)
  ///     │       │           └──  t_2 [8] [CPU] (Leaf: Input)
  ///     │       └──  t_7 [1] [CPU] (Leaf: Input)
  ///     └──  t_180012 [1] [CPU] (Leaf: Input)
  /// ```
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
      branchPrefix = '$branchPrefix└── ';
    } else {
      branchPrefix = '$branchPrefix├── ';
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
          nextPrefix = '$nextPrefix    ';
        } else {
          nextPrefix = '$nextPrefix│   ';
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
            nextPrefix = '$nextPrefix    ';
          } else {
            nextPrefix = '$nextPrefix│   ';
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
      branchPrefix = '$branchPrefix└── ';
    } else {
      branchPrefix = '$branchPrefix├── ';
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
        shapeStr = '$shapeStr, ';
      }
    }
    shapeStr = '$shapeStr]';

    if (currentTensor.creator == null) {
      Logger.green('$tId $shapeStr [GPU] (Leaf: VRAM Input)', prefix: branchPrefix);

      // If this VRAM input maps to a CPU tensor, bridge back to the CPU graph
      if (boundaryMap.containsKey(tId)) {
        Tensor cpuSource = boundaryMap[tId]!;
        String nextPrefix = prefix;
        if (isLast) {
          nextPrefix = '$nextPrefix    ';
        } else {
          nextPrefix = '$nextPrefix│   ';
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
          nextPrefix = '$nextPrefix    ';
        } else {
          nextPrefix = '$nextPrefix│   ';
        }

        _buildGPUGraphString(inputs[i], nextPrefix, isLastChild, visited, boundaryMap);
      }
    }
  }
}
