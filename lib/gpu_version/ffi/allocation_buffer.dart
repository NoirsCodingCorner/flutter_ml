import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import '/full_library.dart';

enum AllocOp {
  loadEmpty,
  loadData,
  initRandom,
  free,
}

class AllocEntry {
  String id;
  List<int> shape;
  AllocOp op;
  Float32List? data;
  List<double> params;
  String? sourceOp;
  bool isGrad;

  AllocEntry(
    this.id,
    this.shape,
    this.op, {
    this.data,
    this.params = const <double>[],
    this.sourceOp,
    this.isGrad = false,
  });
}

class AllocBuffer {
  List<AllocEntry> entries = <AllocEntry>[];

  void putEmpty(String id, List<int> shape,
      {String? sourceOp, bool isGrad = false}) {
    entries.add(AllocEntry(
      id,
      shape,
      AllocOp.loadEmpty,
      sourceOp: sourceOp,
      isGrad: isGrad,
    ));
  }

  void putData(String id, List<int> shape, Float32List data,
      {String? sourceOp}) {
    entries.add(
      AllocEntry(id, shape, AllocOp.loadData, data: data, sourceOp: sourceOp),
    );
  }

  void putRandomUniform(String id, List<int> shape, double scale, int seed,
      {String? sourceOp}) {
    entries.add(AllocEntry(
      id,
      shape,
      AllocOp.initRandom,
      params: <double>[scale, seed.toDouble()],
      sourceOp: sourceOp,
    ));
  }

  void putFree(String id, {String? sourceOp}) {
    entries.add(AllocEntry(id, <int>[], AllocOp.free, sourceOp: sourceOp));
  }

  void clear() {
    entries.clear();
  }

  int _elementCount(List<int> shape) {
    int count = 1;
    for (int d = 0; d < shape.length; d = d + 1) {
      count = count * shape[d];
    }
    return count;
  }

  String _formatBytes(double bytes) {
    if (bytes >= 1073741824.0) {
      return '${(bytes / 1073741824.0).toStringAsFixed(2)} GB';
    }
    if (bytes >= 1048576.0) {
      return '${(bytes / 1048576.0).toStringAsFixed(2)} MB';
    }
    if (bytes >= 1024.0) {
      return '${(bytes / 1024.0).toStringAsFixed(2)} KB';
    }
    return '${bytes.toStringAsFixed(0)} B';
  }

  /// Logs a compact summary of the [AllocBuffer], including total memory (data + IDs),
  /// tensor count, median size, and unique source operations.
  /// When [verbose] is true, also logs every individual entry and the full per-operation breakdown.
  void logBuffer({bool verbose = false}) {
    int totalDataBytes = 0;
    int totalIdBytes = 0;
    int emptyCount = 0;
    int dataCount = 0;
    int randomCount = 0;
    int freeCount = 0;

    List<int> tensorSizesBytes = <int>[];
    List<int> tensorElementCounts = <int>[];
    Map<String, int> sourceOpCounts = <String, int>{};
    Map<String, int> sourceOpBytes = <String, int>{};

    int maxIdLen = 20;
    int maxShapeLen = 20;
    int maxSourceLen = 40;

    for (int i = 0; i < entries.length; i = i + 1) {
      AllocEntry entry = entries[i];
      if (entry.id.length + 2 > maxIdLen) {
        maxIdLen = entry.id.length + 2;
      }
      String shapeStr = entry.shape.toString();
      if (shapeStr.length + 2 > maxShapeLen) {
        maxShapeLen = shapeStr.length + 2;
      }
      String src = entry.sourceOp ?? 'unknown';
      if (src.length + 2 > maxSourceLen) {
        maxSourceLen = src.length + 2;
      }
    }

    String divider =
        '====================================================================================================';
    String subDivider =
        '----------------------------------------------------------------------------------------------------';

    if (verbose) {
      Logger.cyan(divider, prefix: '');
      Logger.cyan('ALLOCATION BUFFER ENTRIES (VERBOSE)', prefix: '');
      Logger.cyan(divider, prefix: '');
    }

    for (int i = 0; i < entries.length; i = i + 1) {
      AllocEntry entry = entries[i];
      String idLabel = entry.id.padRight(maxIdLen);
      String sourceLabel = entry.sourceOp ?? 'unknown';
      String idx = i.toString().padLeft(5);

      if (entry.op == AllocOp.loadEmpty || entry.op == AllocOp.loadData) {
        int elements = _elementCount(entry.shape);
        int dataBytes = elements * 4;
        int idBytes = entry.id.length;
        int entryBytes = dataBytes + idBytes;

        totalDataBytes = totalDataBytes + dataBytes;
        totalIdBytes = totalIdBytes + idBytes;
        tensorSizesBytes.add(entryBytes);
        tensorElementCounts.add(elements);

        String opTag = '[EMPTY]';
        if (entry.op == AllocOp.loadEmpty) {
          emptyCount = emptyCount + 1;
        } else {
          dataCount = dataCount + 1;
          opTag = '[DATA] ';
        }

        int prevCount = sourceOpCounts[sourceLabel] ?? 0;
        sourceOpCounts[sourceLabel] = prevCount + 1;

        int prevBytes = sourceOpBytes[sourceLabel] ?? 0;
        sourceOpBytes[sourceLabel] = prevBytes + entryBytes;

        if (verbose) {
          String shapeStr = entry.shape.toString().padRight(maxShapeLen);
          String sizeStr = _formatBytes(entryBytes.toDouble()).padLeft(12);
          String idSizeStr = '${idBytes}B id'.padLeft(8);
          if (entry.op == AllocOp.loadData) {
            Logger.green(
              '$opTag [$idx] $idLabel | shape: $shapeStr | $sizeStr ($idSizeStr) | src: $sourceLabel',
              prefix: '',
            );
          } else {
            Logger.blue(
              '$opTag [$idx] $idLabel | shape: $shapeStr | $sizeStr ($idSizeStr) | src: $sourceLabel',
              prefix: '',
            );
          }
        }
      } else if (entry.op == AllocOp.initRandom) {
        randomCount = randomCount + 1;
        if (verbose) {
          String scaleStr = entry.params[0].toStringAsFixed(4);
          int seed = entry.params[1].toInt();
          Logger.cyan(
            '[RAND]  [$idx] $idLabel | scale: $scaleStr, seed: $seed | src: $sourceLabel',
            prefix: '',
          );
        }
      } else if (entry.op == AllocOp.free) {
        freeCount = freeCount + 1;
        if (verbose) {
          Logger.yellow(
            '[FREE]  [$idx] $idLabel | src: $sourceLabel',
            prefix: '',
          );
        }
      }
    }

    int totalBytes = totalDataBytes + totalIdBytes;
    int tensorCount = tensorSizesBytes.length;
    double medianBytes = 0.0;
    double medianElements = 0.0;
    int minBytes = 0;
    int maxBytes = 0;

    if (tensorCount > 0) {
      tensorSizesBytes.sort();
      tensorElementCounts.sort();
      minBytes = tensorSizesBytes[0];
      maxBytes = tensorSizesBytes[tensorCount - 1];

      int mid = tensorCount ~/ 2;
      if (tensorCount % 2 == 1) {
        medianBytes = tensorSizesBytes[mid].toDouble();
        medianElements = tensorElementCounts[mid].toDouble();
      } else {
        medianBytes = (tensorSizesBytes[mid - 1] + tensorSizesBytes[mid]) / 2.0;
        medianElements =
            (tensorElementCounts[mid - 1] + tensorElementCounts[mid]) / 2.0;
      }
    }

    Logger.cyan(divider, prefix: '');
    Logger.cyan('ALLOCATION BUFFER SUMMARY', prefix: '');
    Logger.cyan(divider, prefix: '');
    Logger.green(
      'Total Memory:       ${_formatBytes(totalBytes.toDouble())} ($totalBytes B | Data: ${_formatBytes(totalDataBytes.toDouble())}, IDs: ${_formatBytes(totalIdBytes.toDouble())})',
      prefix: '',
    );
    Logger.green(
      'Tensor Count:       $tensorCount (loadEmpty: $emptyCount, loadData: $dataCount)',
      prefix: '',
    );
    Logger.cyan(
      'Other Ops:          initRandom: $randomCount, free: $freeCount (Total Entries: ${entries.length})',
      prefix: '',
    );
    Logger.blue(
      'Median Size:        ${_formatBytes(medianBytes)} (${medianElements.toStringAsFixed(1)} elements)',
      prefix: '',
    );
    Logger.blue(
      'Min / Max Size:     ${_formatBytes(minBytes.toDouble())} / ${_formatBytes(maxBytes.toDouble())}',
      prefix: '',
    );
    Logger.blue(
      'Source Operations:  ${sourceOpCounts.length} unique',
      prefix: '',
    );

    if (verbose) {
      Logger.cyan(subDivider, prefix: '');
      Logger.cyan('Source Operations Breakdown (sorted by memory):',
          prefix: '');

      List<String> sortedOps = sourceOpCounts.keys.toList();
      sortedOps.sort((String a, String b) =>
          sourceOpBytes[b]!.compareTo(sourceOpBytes[a]!));

      for (int i = 0; i < sortedOps.length; i = i + 1) {
        String opName = sortedOps[i];
        int count = sourceOpCounts[opName]!;
        int bytes = sourceOpBytes[opName]!;
        double share = totalBytes > 0 ? (bytes / totalBytes) * 100.0 : 0.0;

        String nameCol = opName.padRight(maxSourceLen);
        String countCol = '${count}x'.padLeft(6);
        String sizeCol = _formatBytes(bytes.toDouble()).padLeft(12);
        String pctCol = '${share.toStringAsFixed(1)}%'.padLeft(7);

        Logger.blue('  $nameCol | $countCol | $sizeCol ($pctCol)', prefix: '');
      }
    }

    Logger.cyan(divider, prefix: '');
  }

  /// Executes all recorded allocations and initializations on [GPUEngine].
  void execute() {
    for (int i = 0; i < entries.length; i = i + 1) {
      AllocEntry entry = entries[i];

      if (entry.op == AllocOp.free) {
        GPUEngine.free(entry.id);
        continue;
      }

      if (entry.op == AllocOp.initRandom) {
        GPUEngine.initRandomUniform(
          entry.id,
          entry.params[0],
          entry.params[1].toInt(),
        );
        continue;
      }

      int count = _elementCount(entry.shape);

      using((Arena arena) {
        Pointer<Float> ptr = arena<Float>(count);
        if (entry.op == AllocOp.loadData && entry.data != null) {
          Float32List view = ptr.asTypedList(count);
          view.setAll(0, entry.data!);
        }
        GPUEngine.load(entry.id, ptr, entry.shape);
      });
    }
  }

  /// Frees all tensors that were allocated in this buffer in reverse order.
  void freeAll() {
    for (int i = entries.length - 1; i >= 0; i = i - 1) {
      AllocEntry entry = entries[i];
      if (entry.op == AllocOp.loadEmpty || entry.op == AllocOp.loadData) {
        GPUEngine.free(entry.id);
      }
    }
  }

  /// Zeros all grads of all entries that could be found.
  void compileZeroGradTape(CommandBuffer tape) {
    for (int i = 0; i < entries.length; i = i + 1) {
      AllocEntry entry = entries[i];
      if (entry.isGrad == true) {
        tape.putInt(OP_ZERO_GRAD);
        tape.putString(entry.id);
      }
    }
  }
}
