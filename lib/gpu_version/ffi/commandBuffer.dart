import 'dart:typed_data';
import 'dart:convert';

class CommandBuffer {
  int offset = 0;
  int _capacity = 1024;
  ByteData _buffer = ByteData(1024);

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

  void putInt(int value) {
    _expand(4);
    _buffer.setInt32(offset, value, Endian.host);
    offset = offset + 4;
  }

  void putFloat(double value) {
    _expand(4);
    _buffer.setFloat32(offset, value, Endian.host);
    offset = offset + 4;
  }

  void putBool(bool value) {
    _expand(1);
    _buffer.setInt8(offset, value ? 1 : 0);
    offset = offset + 1;
  }

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

  // Returns a copy of the active portion of the buffer.
  Uint8List bytes() {
    Uint8List tape = Uint8List(offset);
    for (int i = 0; i < offset; i = i + 1) {
      tape[i] = _buffer.getUint8(i);
    }
    return tape;
  }

  void clear() {
    offset = 0;
  }
}