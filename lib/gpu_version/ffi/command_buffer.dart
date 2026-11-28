import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import '/full_library.dart';

/// CommandBuffer in this library is used as a tape to record actions for the native backend to execute.
/// Commands are stored as a continuous stream of raw bytes.
/// Each command consists of two parts:
///
/// OpCode (4 bytes): storing which method should be executed on the backend
///
/// Arguments: providing additional arguments for the operation.
/// Example:
/// ```dart
///  [OpCode (4 bytes)] [Len A (2 bytes)] [Text A] [Len B (2 bytes)] [Text B] [Len Out (2 bytes)] [Text Out]
/// ```
/// Usually CommandBuffers do not need to be filled manually but are filled by providing them to [GPUTensor]- operations as arguments.
class CommandBuffer {
  static Random _defaultRng = Random();

  int offset = 0;
  int _capacity = 1024;
  ByteData _buffer = ByteData(1024);
  Map<String, List<int>> patchOffsets = <String, List<int>>{};

  /// Method to expand the capacity of the buffer. Inserting new arguments or values increases bufferSize.
  /// Setting this value manually is not advised.
  void _expand(int required) {
    if (offset + required <= _capacity) return;

    while (offset + required > _capacity) {
      _capacity = _capacity * 2;
    }

    ByteData bigger = ByteData(_capacity);
    for (int i = 0; i < offset; i = i + 1) {
      bigger.setUint8(i, _buffer.getUint8(i));
    }
    _buffer = bigger;
  }

  void _registerPatch(String key, int patchOffset) {
    List<int>? offsets = patchOffsets[key];
    if (offsets == null) {
      offsets = <int>[];
      patchOffsets[key] = offsets;
    }
    offsets.add(patchOffset);
  }

  /// Inserts an integer into the buffer.
  void putInt(int value) {
    _expand(4);
    _buffer.setInt32(offset, value, Endian.host);
    offset = offset + 4;
  }

  /// Inserts a float into the buffer.
  void putFloat(double value) {
    _expand(4);
    _buffer.setFloat32(offset, value, Endian.host);
    offset = offset + 4;
  }

  /// Inserts a bool into the buffer.
  void putBool(bool value) {
    _expand(1);
    _buffer.setInt8(offset, value ? 1 : 0);
    offset = offset + 1;
  }

  /// Inserts a given String into the buffer via UTF-8 encoding.
  void putString(String text) {
    Uint8List encoded = utf8.encode(text);
    int count = encoded.length;

    // We reserve 2 bytes for the length prefix (Uint16)
    _expand(2 + count);

    _buffer.setUint16(offset, count, Endian.host);
    offset = offset + 2;

    for (int i = 0; i < count; i = i + 1) {
      _buffer.setUint8(offset, encoded[i]);
      offset = offset + 1;
    }
  }

  /// Inserts an integer and records its byte offset under [key] for later in-place patching.
  void putPatchableInt(String key, int value) {
    _registerPatch(key, offset);
    putInt(value);
  }

  /// Inserts a float and records its byte offset under [key] for later in-place patching.
  void putPatchableFloat(String key, double value) {
    _registerPatch(key, offset);
    putFloat(value);
  }

  /// Inserts a bool and records its byte offset under [key] for later in-place patching.
  void putPatchableBool(String key, bool value) {
    _registerPatch(key, offset);
    putBool(value);
  }

  /// Overwrites all 4-byte ints registered under [key] in place.
  void patchInt(String key, int value) {
    List<int>? offsets = patchOffsets[key];
    if (offsets == null) return;

    for (int i = 0; i < offsets.length; i = i + 1) {
      _buffer.setInt32(offsets[i], value, Endian.host);
    }
  }

  /// Overwrites all 4-byte floats registered under [key] in place.
  void patchFloat(String key, double value) {
    List<int>? offsets = patchOffsets[key];
    if (offsets == null) return;

    for (int i = 0; i < offsets.length; i = i + 1) {
      _buffer.setFloat32(offsets[i], value, Endian.host);
    }
  }

  /// Overwrites all 1-byte bools registered under [key] in place.
  void patchBool(String key, bool value) {
    List<int>? offsets = patchOffsets[key];
    if (offsets == null) return;

    int byteValue = value ? 1 : 0;
    for (int i = 0; i < offsets.length; i = i + 1) {
      _buffer.setInt8(offsets[i], byteValue);
    }
  }

  /// Overwrites each 4-byte int registered under [key] with an independent random value.
  void patchRandomInts(String key, {Random? rng, int max = 1000000}) {
    List<int>? offsets = patchOffsets[key];
    if (offsets == null) return;

    Random random = rng ?? _defaultRng;
    for (int i = 0; i < offsets.length; i = i + 1) {
      _buffer.setInt32(offsets[i], random.nextInt(max), Endian.host);
    }
  }

  /// Returns a zero-copy view of the active portion of the buffer.
  /// Primarily used to run CommandBuffers on the GPU:
  /// ```dart
  ///  GPUEngine.run(commandBuffer.bytes());
  /// ```
  Uint8List bytes() {
    return Uint8List.view(_buffer.buffer, 0, offset);
  }

  /// Resets the storage offset to 0 and clears registered patch offsets.
  void clear() {
    offset = 0;
    patchOffsets.clear();
  }
}