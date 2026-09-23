import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

/// Targeted device architecture
enum Target {
  android_arm64,
  android_x86_64,
  cuda,
  auto // Added a recommended 'auto' option
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

/// FFI- Binding for retrieving the ids of all tensors currently on the GPU.
typedef NativeGetNames = Pointer<Utf8> Function(Int64);
typedef DartGetNames   = Pointer<Utf8> Function(int);

/// FFI- Binding for freeing the made list of tensor names
typedef NativeFreeStr  = Void Function(Pointer<Utf8>);
typedef DartFreeStr    = void Function(Pointer<Utf8>);

/// FFI- Advanced operation to allow direct pointer access.
typedef NativeAddPointers = Void Function(Pointer<Float>, Pointer<Float>, Int32);
typedef DartAddPointers   = void Function(Pointer<Float>, Pointer<Float>, int);

/// FFI- Initialise a tensor with a random initialization
typedef NativeInitRandom = Void Function(Int64, Pointer<Utf8>, Float, Int32);
typedef DartInitRandom   = void Function(int, Pointer<Utf8>, double, int);


/// The Cuda Engine in responsible for communication with the native runtimes as well as managing the communication between dart and its native FFI-bindings.
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

  /// Initialises the [GPUEngine].
  /// [debug] enables full printout of every interaction for in debugging.
  static Future<void> initialize({bool debug = false, Target target = Target.auto}) async {
    String libName = "";

    // 1. Resolve Library Name based on Platform & Target
    if (Platform.isAndroid) {
      if (target == Target.android_arm64) {
        libName = "libandroid_arm64.so";
      } else if (target == Target.android_x86_64) {
        libName = "libandroid_x86_64.so";
      } else {
        libName = "libandroid_arm64.so";
      }
    } else if (Platform.isWindows) {
      libName = "cuda_executor.dll";
    } else if (Platform.isLinux) {
      libName = "libcuda_executor.so";
    } else {
      throw UnsupportedError("GPUEngine currently only supports Android, Windows, and Linux.");
    }

    DynamicLibrary dylib;

    // 2. Open Library
    try {
      dylib = DynamicLibrary.open(libName);
      print("GPUEngine: Successfully loaded $libName backend via standard OS paths.");
    } catch (e) {
      // 3. Fallback for Local Development & Testing
      print("Standard load failed, attempting local path fallback for testing...");
      try {
        String currentPath = Directory.current.path;

        if (currentPath.endsWith('example') || currentPath.endsWith('example\\') || currentPath.endsWith('example/')) {
          currentPath = Directory(currentPath).parent.path;
        }

        String fallbackPath = "";
        if (Platform.isWindows) {
          fallbackPath = "$currentPath\\windows\\bin\\$libName";
        } else if (Platform.isLinux) {
          fallbackPath = "$currentPath/linux/bin/$libName";
        } else {
          fallbackPath = "$currentPath/android/src/main/jniLibs/arm64-v8a/$libName";
        }

        dylib = DynamicLibrary.open(fallbackPath);
        print("GPUEngine: Successfully loaded via local fallback path.");
      } catch (fallbackError) {
        print("CRITICAL FFI ERROR: Failed to open library $libName.");
        print("Primary Error: $e");
        print("Fallback Error: $fallbackError");
        rethrow;
      }
    }

    // 4. Bind Functions
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

  /// Allocates and transfers tensor data from host to device (GPU).
  static void load(String name, Pointer<Float> data, List<int> shape) {
    Pointer<Utf8>  nName  = name.toNativeUtf8();
    Pointer<Int32> nShape = calloc<Int32>(shape.length);

    Int32List view = nShape.asTypedList(shape.length);
    for (int i = 0; i < shape.length; i++) {
      view[i] = shape[i];
    }

    _load(handle, nName, data, shape.length, nShape);

    calloc.free(nName);
    calloc.free(nShape);
  }

  /// Transfers a tensors data from device(GPU) to host.
  static void retrieve(String name, Pointer<Float> destination) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _retrieve(handle, nName, destination);
    calloc.free(nName);
  }

  /// Passes a compiled [CommandBuffer] tape to the native engine and executes it.
  static void run(Uint8List tapeBytes) {
    Pointer<Uint8> nTape = calloc<Uint8>(tapeBytes.length);
    Uint8List      view  = nTape.asTypedList(tapeBytes.length);

    for (int i = 0; i < tapeBytes.length; i++) {
      view[i] = tapeBytes[i];
    }

    _run(handle, nTape, tapeBytes.length);
    calloc.free(nTape);
  }

  /// Frees a given tensors allocation on the device(GPU).
  static void free(String name) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _freeTensor(handle, nName);
    calloc.free(nName);
  }

  /// Allows a direct memory copy from [src] to [dest] for faster transfer.
  static void addPointers(Pointer<Float> dest, Pointer<Float> src, int length) {
    _addPointers(dest, src, length);
  }

  /// Retrieves all tensor names currently allocated on the GPU.
  static List<String> getTensorNames() {
    Pointer<Utf8> cStringPtr = _getTensorNames(handle);

    if (cStringPtr == nullptr) {
      return <String>[];
    }

    String combinedNames = cStringPtr.toDartString();
    _freeString(cStringPtr);

    if (combinedNames.isEmpty) {
      return <String>[];
    }

    List<String> rawNames = combinedNames.split(',');
    List<String> cleanNames = <String>[];

    for (int i = 0; i < rawNames.length; i++) {
      cleanNames.add(rawNames[i].trim());
    }

    return cleanNames;
  }

  /// Deletes the current instance of the GPUEngine and frees all allocated memory.
  static void dispose() {
    if (handle != 0) {
      _free(handle);
      handle = 0;
    }
  }

  /// Fills the tensor [name] with uniformly distributed numbers scaled with [scale].
  static void initRandomUniform(String name, double scale, int seed) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _initRandom(handle, nName, scale, seed);
    calloc.free(nName);
  }
}