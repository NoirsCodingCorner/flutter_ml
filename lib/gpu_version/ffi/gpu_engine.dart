import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '/full_library.dart';

/// Targeted device architecture
enum Target{
  androidArm64,
  androidX8664,
  cuda
}
/// FFI- Binding for creating a tensor on the GPU
typedef NativeCreate   = Int64 Function(Bool);
typedef DartCreate     = int Function(bool);

/// FFI- Binding for freeing/deleting the GPU backend
typedef NativeFree     = Void Function(Int64);
typedef DartFree       = void Function(int);

/// FFI- Binding for pushing a tensor to the GPU
typedef NativeLoad     = Void Function(Int64, Pointer<Utf8>, Pointer<Float>, Int32, Pointer<Int32>);
typedef DartLoad       = void Function(int, Pointer<Utf8>, Pointer<Float>, int, Pointer<Int32>);

/// FFI- Binding for pulling a tensor from the GPU
typedef NativeRetrieve = Void Function(Int64, Pointer<Utf8>, Pointer<Float>);
typedef DartRetrieve   = void Function(int, Pointer<Utf8>, Pointer<Float>);

/// FFI- Binding for running a given byteList of commands (usually via a [CommandBuffer])
typedef NativeRun      = Void Function(Int64, Pointer<Uint8>, Int32);
typedef DartRun        = void Function(int, Pointer<Uint8>, int);

/// FFI- Binding for freeing a tensor on the GPU
typedef NativeFreeT    = Void Function(Int64, Pointer<Utf8>);
typedef DartFreeT      = void Function(int, Pointer<Utf8>);

/// FFI- Binding for retrieving the ids of all tensors currently on the GPU. This creates a List of Strings that is stored in the backend and has to be deleted again
typedef NativeGetNames = Pointer<Utf8> Function(Int64);
typedef DartGetNames   = Pointer<Utf8> Function(int);

/// FFI- Binding for freeing the made list of tensor names
typedef NativeFreeStr  = Void Function(Pointer<Utf8>);
typedef DartFreeStr    = void Function(Pointer<Utf8>);

/// FFI- Advanced operation to allow direct pointer access. Adds the content of a source content into a destination pointer with a given length
typedef NativeAddPointers = Void Function(Pointer<Float>, Pointer<Float>, Int32);
typedef DartAddPointers   = void Function(Pointer<Float>, Pointer<Float>, int);

/// FFI- Initialise a tensor with a random initialization
typedef NativeInitRandom = Void Function(Int64, Pointer<Utf8>, Float, Int32);
typedef DartInitRandom   = void Function(int, Pointer<Utf8>, double, int);



/// The Cuda Engine in responsible for communication with the native runtimes as well as managing the communication between dart and its native FFI-bindings.
/// In order to use the GPU acceleration, at the very beginning of the program it is required to call [initialize] with the [Target] provided to load the relevant binary.
/// This class is to be used globally as a sole instance of communication and data transfer.
/// Its [handle] acts as the id of the native executor instance. If there is currently no instance running this value will be 0.
/// Currently supported are WebGPU with `android_arm64`, `android_x86_64` and `cuda`from version 1.12.1 (2026-09).
/// Use [dispose] to delete the currently running instance of GPUEngine and free all memory again.

class GPUEngine {

  /// Id of the currently running GPUEngine. If its value is 0, no engine is running.
  static int handle = 0;

  static late DartCreate      _create;
  static late DartFree        _free;
  static late DartLoad        _load;
  static late DartRetrieve    _retrieve;
  static late DartRun         _run;
  static late DartFreeT       _freeTensor;
  static late DartGetNames    _getTensorNames;
  static late DartFreeStr     _freeString;
  static late DartAddPointers _addPointers;
  static late DartInitRandom  _initRandom;

  /// Initialises the [GPUEngine]. If no target platform is provided initialising with `cuda` is attempted.
  /// [debug] enables full printout of every interaction for in debugging. Initialize has to be called at least once before using the GPUEngine.
  static Future<void> initialize({bool debug = false, Target target = Target.cuda}) async {
    String libName = "";

    if (target == Target.cuda) {
      libName = "cuda_executor.dll";
    } else if (target == Target.androidArm64) {
      libName = "libandroid_arm64.so";
    } else if (target == Target.androidX8664) {
      libName = "libandroid_x86_64.so";
    }

    DynamicLibrary dylib;

    try {
      // Windows automatically checks the directory of the executable for the DLL and its dependencies.
      if (Platform.isAndroid || Platform.isWindows) {
        dylib = DynamicLibrary.open(libName);
      } else {
        throw Exception("Unsupported platform");
      }

      Logger.green("GPUEngine: Successfully loaded ${target.name} backend.");

    } catch (e) {
      Logger.red("CRITICAL FFI ERROR: Failed to open library $libName. Error: $e");
      rethrow;
    }

    _create         = dylib.lookupFunction<NativeCreate, DartCreate>('create_executor');
    _free           = dylib.lookupFunction<NativeFree, DartFree>('free_executor');
    _load           = dylib.lookupFunction<NativeLoad, DartLoad>('load_tensor_h2d');
    _retrieve       = dylib.lookupFunction<NativeRetrieve, DartRetrieve>('retrieve_tensor_d2h_into');
    _run            = dylib.lookupFunction<NativeRun, DartRun>('run_tape');
    _freeTensor     = dylib.lookupFunction<NativeFreeT, DartFreeT>('free_tensor');
    _getTensorNames = dylib.lookupFunction<NativeGetNames, DartGetNames>('get_tensor_names');
    _freeString     = dylib.lookupFunction<NativeFreeStr, DartFreeStr>('free_string');
    _addPointers    = dylib.lookupFunction<NativeAddPointers, DartAddPointers>('add_pointers');
    _initRandom     = dylib.lookupFunction<NativeInitRandom, DartInitRandom>('init_random_uniform');

    handle = _create(debug);
  }

  /// Allocates and transfers tensor data from host to device (GPU). Requires the tensors name, data and shape.
  static void load(String name, Pointer<Float> data, List<int> shape) {
    using((Arena arena) {
      Pointer<Utf8>  nName  = name.toNativeUtf8(allocator: arena);
      Pointer<Int32> nShape = arena<Int32>(shape.length);
      Int32List view = nShape.asTypedList(shape.length);
      for (int i = 0; i < shape.length; i = i + 1) {
        view[i] = shape[i];
      }
      _load(handle, nName, data, shape.length, nShape);
    });
  }

  /// Transfers a tensors data from device(GPU) to host. Requires the tensors name and transfer destination.
  static void retrieve(String name, Pointer<Float> destination) {
    using((Arena arena) {
      Pointer<Utf8> nName = name.toNativeUtf8(allocator: arena);
      _retrieve(handle, nName, destination);
    });
  }

  /// Passes a compiled [CommandBuffer] tape to the native engine and executes it. Bytes are copied into memory before dispatch.
  static void run(Uint8List tapeBytes) {
    using((Arena arena) {
      Pointer<Uint8> nTape = arena<Uint8>(tapeBytes.length);
      Uint8List view = nTape.asTypedList(tapeBytes.length);

      // Fast path memory copy (C-optimized) instead of manual loop
      view.setAll(0, tapeBytes);

      _run(handle, nTape, tapeBytes.length);
    });
  }

  /// Frees a given tensors allocation on the device(GPU).
  static void free(String name) {
    using((Arena arena) {
      Pointer<Utf8> nName = name.toNativeUtf8(allocator: arena);
      _freeTensor(handle, nName);
    });
  }

  /// Allows a direct memory copy from [src] to [dest] for faster transfer.
  static void addPointers(Pointer<Float> dest, Pointer<Float> src, int length) {
    _addPointers(dest, src, length);
  }

  /// Retrieves all tensor names currently allocated on the GPU. Allocates memory to store the String list natively and deletes it before returning its value,
  static List<String> getTensorNames() {
    Pointer<Utf8> cStringPtr = _getTensorNames(handle);

    if (cStringPtr == nullptr) {
      return <String>[];
    }

    String combinedNames = cStringPtr.toDartString();

    // _freeString remains manual because memory was allocated inside C++, not Dart.
    _freeString(cStringPtr);

    if (combinedNames.isEmpty) {
      return <String>[];
    }

    List<String> rawNames = combinedNames.split(',');
    List<String> cleanNames = <String>[];

    for (int i = 0; i < rawNames.length; i = i + 1) {
      cleanNames.add(rawNames[i].trim());
    }

    return cleanNames;
  }

  /// Deletes the current instance of the GPUEngine and frees all allocated memory.
  static void dispose() {
    _free(handle);
    handle = 0;
  }

  /// Fills the tensor [name] with uniformly distributed numbers scaled with [scale].
  static void initRandomUniform(String name, double scale, int seed) {
    using((Arena arena) {
      Pointer<Utf8> nName = name.toNativeUtf8(allocator: arena);
      _initRandom(handle, nName, scale, seed);
    });
  }
}