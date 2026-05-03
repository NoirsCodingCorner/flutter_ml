import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import 'commandBuffer.dart';

typedef NativeCreate   = Int64 Function(Bool);
typedef DartCreate     = int Function(bool);

typedef NativeFree     = Void Function(Int64);
typedef DartFree       = void Function(int);

typedef NativeLoad     = Void Function(Int64, Pointer<Utf8>, Pointer<Float>, Int32, Pointer<Int32>);
typedef DartLoad       = void Function(int, Pointer<Utf8>, Pointer<Float>, int, Pointer<Int32>);

typedef NativeRetrieve = Void Function(Int64, Pointer<Utf8>, Pointer<Float>);
typedef DartRetrieve   = void Function(int, Pointer<Utf8>, Pointer<Float>);

typedef NativeRun      = Void Function(Int64, Pointer<Uint8>, Int32);
typedef DartRun        = void Function(int, Pointer<Uint8>, int);

typedef NativeFreeT    = Void Function(Int64, Pointer<Utf8>);
typedef DartFreeT      = void Function(int, Pointer<Utf8>);

typedef NativeGetNames = Pointer<Utf8> Function(Int64);
typedef DartGetNames   = Pointer<Utf8> Function(int);

typedef NativeFreeStr  = Void Function(Pointer<Utf8>);
typedef DartFreeStr    = void Function(Pointer<Utf8>);

// NEW: Add Pointers Typedefs
typedef NativeAddPointers = Void Function(Pointer<Float>, Pointer<Float>, Int32);
typedef DartAddPointers   = void Function(Pointer<Float>, Pointer<Float>, int);

typedef NativeInitRandom = Void Function(Int64, Pointer<Utf8>, Float, Int32);
typedef DartInitRandom = void Function(int, Pointer<Utf8>, double, int);

class CudaEngine {
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
  static late DartInitRandom _initRandom;

  static void initialize({bool debug = false}) {
    String path = Platform.isWindows
        ? 'cuda_engine/cmake-build-debug/cuda_executor.dll'
        : './libcuda_executor.so';

    DynamicLibrary dylib = DynamicLibrary.open(path);

    _create         = dylib.lookupFunction<NativeCreate, DartCreate>('create_executor');
    _free           = dylib.lookupFunction<NativeFree, DartFree>('free_executor');
    _load           = dylib.lookupFunction<NativeLoad, DartLoad>('load_tensor_h2d');
    _retrieve       = dylib.lookupFunction<NativeRetrieve, DartRetrieve>('retrieve_tensor_d2h_into');
    _run            = dylib.lookupFunction<NativeRun, DartRun>('run_tape');
    _freeTensor     = dylib.lookupFunction<NativeFreeT, DartFreeT>('free_tensor');
    _getTensorNames = dylib.lookupFunction<NativeGetNames, DartGetNames>('get_tensor_names');
    _freeString     = dylib.lookupFunction<NativeFreeStr, DartFreeStr>('free_string');

    // NEW: Bind the native memory addition function
    _addPointers    = dylib.lookupFunction<NativeAddPointers, DartAddPointers>('add_pointers');
    _initRandom     = dylib.lookupFunction<NativeInitRandom, DartInitRandom>('init_random_uniform');
    handle = _create(debug);
  }

  static void load(String name, Pointer<Float> data, List<int> shape) {
    Pointer<Utf8>  nName  = name.toNativeUtf8();
    Pointer<Int32> nShape = calloc<Int32>(shape.length);

    Int32List view = nShape.asTypedList(shape.length);
    for (int i = 0; i < shape.length; i = i + 1) {
      view[i] = shape[i];
    }

    _load(handle, nName, data, shape.length, nShape);

    calloc.free(nName);
    calloc.free(nShape);
  }

  static void retrieve(String name, Pointer<Float> destination) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _retrieve(handle, nName, destination);
    calloc.free(nName);
  }

  static void run(Uint8List tapeBytes) {
    Pointer<Uint8> nTape = calloc<Uint8>(tapeBytes.length);
    Uint8List      view  = nTape.asTypedList(tapeBytes.length);

    for (int i = 0; i < tapeBytes.length; i = i + 1) {
      view[i] = tapeBytes[i];
    }

    _run(handle, nTape, tapeBytes.length);
    calloc.free(nTape);
  }

  static void free(String name) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _freeTensor(handle, nName);
    calloc.free(nName);
  }

  // NEW: Expose the instant memory addition to Dart
  static void addPointers(Pointer<Float> dest, Pointer<Float> src, int length) {
    _addPointers(dest, src, length);
  }

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

    for (int i = 0; i < rawNames.length; i = i + 1) {
      cleanNames.add(rawNames[i].trim());
    }

    return cleanNames;
  }

  static void dispose() {
    _free(handle);
    handle = 0;
  }

  static void initRandomUniform(String name, double scale, int seed) {
    Pointer<Utf8> nName = name.toNativeUtf8();
    _initRandom(handle, nName, scale, seed);
    calloc.free(nName);
  }
}