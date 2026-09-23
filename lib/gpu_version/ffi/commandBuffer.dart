import 'dart:typed_data';
import 'dart:convert';

import 'package:flutter_ml/full_library.dart';

/// CommandBuffer in this library is used as a tape to record actions for the native backend to execute.
/// Commands are stored as a continuous stream of raw bytes.
/// Each command constists of two parts:
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
  int offset = 0;
  int _capacity = 1024;
  ByteData _buffer = ByteData(1024);

  /// Method to expand the capacity of the buffer. Inserting new arguments or values increases bufferSize.
  /// Setting this value manually is not advised
  void _expand(int required) {
    if (offset + required > _capacity) {
      _capacity = _capacity * 2;
      ByteData bigger = ByteData(_capacity);
      for (int i = 0; i < offset; i = i + 1) {
        bigger.setUint8(i, _buffer.getUint8(i));
      }
      _buffer = bigger;
    }
  }
  /// Inserts an integer into the buffer
  void putInt(int value) {
    _expand(4);
    _buffer.setInt32(offset, value, Endian.host);
    offset = offset + 4;
  }

  /// Inserts a float into the buffer
  void putFloat(double value) {
    _expand(4);
    _buffer.setFloat32(offset, value, Endian.host);
    offset = offset + 4;
  }

  /// Inserts a bool into the buffer
  void putBool(bool value) {
    _expand(1);
    _buffer.setInt8(offset, value ? 1 : 0);
    offset = offset + 1;
  }

  /// Inserts a given String into the buffer vie encoding over uft-8
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

  /// Returns a copy of the active portion of the buffer.
  /// Primarily used to decode the buffer into a Uint8List to run CommandBuffers on the GPU:
  /// ```dart
  ///  GPUEngine.run(commandBuffer.bytes())
  /// ```
  Uint8List bytes() {
    Uint8List tape = Uint8List(offset);
    for (int i = 0; i < offset; i = i + 1) {
      tape[i] = _buffer.getUint8(i);
    }
    return tape;
  }

  /// sets the storage offset to 0
  void clear() {
    offset = 0;
  }
}