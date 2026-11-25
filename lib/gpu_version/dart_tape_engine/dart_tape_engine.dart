import 'dart:ffi';
import 'dart:typed_data';
import 'dart_cpu_manager.dart';
import 'dart_graph_executor.dart';

class DartTapeEngine {
  static DartCpuManager? _manager;
  static DartGraphExecutor? _executor;

  static void initialize({bool debug = false}) {
    _manager = DartCpuManager();
    _executor = DartGraphExecutor(_manager!, debug: debug);
  }

  static void dispose() {
    if (_manager != null) {
      _manager!.clear();
    }
    _executor = null;
    _manager = null;
  }

  static void load(String name, Pointer<Float> data, List<int> shape) {
    DartCpuManager manager = _manager!;

    if (manager.exists(name)) {
      DartCpuElement existing = manager.getElement(name);
      bool shapeMismatch = false;

      if (existing.sizes.length != shape.length) {
        shapeMismatch = true;
      } else {
        for (int i = 0; i < shape.length; i = i + 1) {
          if (existing.sizes[i] != shape[i]) {
            shapeMismatch = true;
            break;
          }
        }
      }

      if (shapeMismatch) {
        manager.free(name);
        manager.allocate(name, shape);
      }
    } else {
      manager.allocate(name, shape);
    }

    manager.copyHostToDevice(name, data);
  }

  static void retrieve(String name, Pointer<Float> destination) {
    _manager!.retrieveInto(name, destination);
  }

  static void run(Uint8List tapeBytes) {
    _executor!.runTape(tapeBytes);
  }

  static void free(String name) {
    if (_manager!.exists(name)) {
      _manager!.free(name);
    }
  }

  static void addPointers(Pointer<Float> dest, Pointer<Float> src, int length) {
    Float32List destView = dest.asTypedList(length);
    Float32List srcView = src.asTypedList(length);
    for (int i = 0; i < length; i = i + 1) {
      destView[i] = destView[i] + srcView[i];
    }
  }

  static List<String> getTensorNames() {
    List<String> names = <String>[];
    for (String key in _manager!.memory.keys) {
      names.add(key);
    }
    return names;
  }

  static int getTensorCount() {
    return _manager!.memory.length;
  }

  static void initRandomUniform(String name, double scale, int seed) {
    _manager!.initRandomUniform(name, scale, seed);
  }
}