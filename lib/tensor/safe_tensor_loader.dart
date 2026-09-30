import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import '../logger.dart';

import 'tensor_gpu.dart';

/// Utility class for reading and deserializing model weights from Hugging Face Safetensors files into [GPUTensor] instances.
/// Parses the 8-byte header length, extracts the JSON metadata header, slices binary tensor chunks,
/// handles matrix transpositions between coordinate standards, and pushes data directly into VRAM.
class SafetensorsLoader {
  /// Parsed JSON metadata header mapping tensor names to data types, shapes, and binary byte offsets.
  Map<String, dynamic> header = <String, dynamic>{};

  /// Raw byte buffer of the entire loaded Safetensors file.
  late Uint8List rawBytes;

  /// Byte index where raw tensor binary data begins following the 8-byte length prefix and JSON header.
  int bufferOffset = 0;

  /// Reads a Safetensors file synchronously from [filePath], decodes the 8-byte header size prefix,
  /// parses the JSON metadata header, and determines [bufferOffset].
  SafetensorsLoader(String filePath) {
    File file = File(filePath);
    rawBytes = file.readAsBytesSync();

    ByteData byteData = ByteData.sublistView(rawBytes);

    int headerSize = byteData.getUint64(0, Endian.little);
    bufferOffset = 8 + headerSize;

    Uint8List headerBytes = Uint8List(headerSize);
    for (int i = 0; i < headerSize; i = i + 1) {
      headerBytes[i] = rawBytes[8 + i];
    }

    String headerString = utf8.decode(headerBytes);
    Map<String, dynamic> parsedJson = json.decode(headerString);
    header = parsedJson;
  }

  /// Returns a list of all tensor keys available in the parsed header, excluding the `__metadata__` block.
  List<String> getAvailableTensors() {
    List<String> keys = header.keys.toList();
    List<String> result = <String>[];

    for (int i = 0; i < keys.length; i = i + 1) {
      if (keys[i] != '__metadata__') {
        result.add(keys[i]);
      }
    }

    return result;
  }

  /// Extracts binary float values from the buffer for all matching keys in [modelTensors],
  /// automatically transposes 2D matrices if row and column dimensions are inverted,
  /// and uploads the resulting values directly to VRAM via [GPUTensor.pushData].
  void loadIntoTensors(Map<String, GPUTensor> modelTensors) {
    List<String> keys = modelTensors.keys.toList();

    for (int i = 0; i < keys.length; i = i + 1) {
      String key = keys[i];

      if (header.containsKey(key) == false) {
        continue;
      }

      GPUTensor tensor = modelTensors[key]!;

      Map<String, dynamic> tensorMeta = header[key];
      List<dynamic> jsonShape = tensorMeta['shape'];
      List<dynamic> offsets = tensorMeta['data_offsets'];

      int startOffset = offsets[0];
      int endOffset = offsets[1];

      int absoluteStart = bufferOffset + startOffset;
      int byteLength = endOffset - startOffset;
      int floatCount = byteLength ~/ 4;

      ByteData tensorData = ByteData.sublistView(rawBytes, absoluteStart, absoluteStart + byteLength);
      List<double> values = <double>[];

      for (int j = 0; j < floatCount; j = j + 1) {
        values.add(tensorData.getFloat32(j * 4, Endian.little));
      }

      if (tensor.shape.length == 2 && jsonShape.length == 2) {
        int hfRows = jsonShape[0];
        int hfCols = jsonShape[1];
        int engineRows = tensor.shape[0];
        int engineCols = tensor.shape[1];

        if (hfRows == engineCols && hfCols == engineRows) {
          List<double> transposed = <double>[];
          for (int k = 0; k < floatCount; k = k + 1) {
            transposed.add(0.0);
          }

          for (int r = 0; r < hfRows; r = r + 1) {
            for (int c = 0; c < hfCols; c = c + 1) {
              int srcIndex = (r * hfCols) + c;
              int destIndex = (c * hfRows) + r;
              transposed[destIndex] = values[srcIndex];
            }
          }
          values = transposed;
        }
      }

      tensor.pushData(values);
    }
  }
}

/// Reads a Safetensors file from [filePath] and logs the name, shape, and data type of every serialized tensor.
void printSafetensorsStructure(String filePath) {
  File file = File(filePath);
  Uint8List rawBytes = file.readAsBytesSync();

  ByteData byteData = ByteData.sublistView(rawBytes);
  int headerSize = byteData.getUint64(0, Endian.little);

  Uint8List headerBytes = Uint8List(headerSize);
  for (int i = 0; i < headerSize; i = i + 1) {
    headerBytes[i] = rawBytes[8 + i];
  }

  String headerString = utf8.decode(headerBytes);
  Map<String, dynamic> header = json.decode(headerString);

  List<String> keys = header.keys.toList();

  for (int i = 0; i < keys.length; i = i + 1) {
    String key = keys[i];

    if (key == '__metadata__') {
      continue;
    }

    Map<String, dynamic> layerInfo = header[key];
    List<dynamic> shape = layerInfo['shape'];
    String dtype = layerInfo['dtype'];

    Logger.log(key);
    Logger.log("$shape");
    Logger.log(dtype);
  }
}

/*
/// Entry point demonstrating the inspection and logging of tensor metadata from a Safetensors file.
void main() {
  printSafetensorsStructure('models/bert/model.safetensors');
}
*/
