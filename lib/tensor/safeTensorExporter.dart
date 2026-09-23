import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import '../logger.dart';

import 'tensor_gpu.dart';

class SafetensorsExporter {
  /// Flattens nested Dart lists into a single 1D array of doubles
  static List<double> _flattenTensor(dynamic value) {
    if (value is double) {
      return <double>[value];
    } else if (value is List) {
      List<double> flat = <double>[];
      for (int i = 0; i < value.length; i = i + 1) {
        flat.addAll(_flattenTensor(value[i]));
      }
      return flat;
    }
    return <double>[];
  }

  /// Exports the VRAM tensors to a standard Hugging Face .safetensors file
  static void saveModel(Map<String, GPUTensor> modelTensors, String outputPath) {
    Logger.log('Starting export to $outputPath...');

    Map<String, dynamic> jsonHeader = <String, dynamic>{};
    BytesBuilder rawBuffer = BytesBuilder();
    int currentOffset = 0;

    // 1. Iterate through all mapped parameters (e.g., 'bert.encoder.layer.0...')
    List<String> keys = modelTensors.keys.toList();

    for (int i = 0; i < keys.length; i = i + 1) {
      String tensorName = keys[i];
      GPUTensor tensor = modelTensors[tensorName]!;

      // Pull the latest weights from the GPU back to the Dart CPU Heap
      tensor.toCpu();

      // Flatten the multidimensional array
      List<double> flatData = _flattenTensor(tensor.value);

      // Convert Dart 64-bit doubles to 32-bit floats for standard HF F32 format
      Float32List f32List = Float32List.fromList(flatData);
      Uint8List byteData = f32List.buffer.asUint8List();

      int byteLength = byteData.length;

      // 2. Build the JSON Metadata for this specific tensor
      jsonHeader[tensorName] = <String, dynamic>{
        "dtype": "F32",
        "shape": tensor.shape,
        "data_offsets": <int>[currentOffset, currentOffset + byteLength]
      };

      // 3. Append to the giant binary buffer
      rawBuffer.add(byteData);
      currentOffset = currentOffset + byteLength;

      Logger.log('Processed: $tensorName (Shape: ${tensor.shape})');
    }

    // Safetensors requires an "__metadata__" block (often used for format info)
    jsonHeader["__metadata__"] = <String, String>{
      "format": "pt"
    };

    // 4. Encode JSON
    String jsonString = jsonEncode(jsonHeader);
    Uint8List jsonBytes = utf8.encode(jsonString);

    // 5. Create the 8-byte little-endian header length prefix
    ByteData headerLengthBytes = ByteData(8);
    headerLengthBytes.setUint64(0, jsonBytes.length, Endian.little);

    // 6. Write everything to disk!
    File file = File(outputPath);
    BytesBuilder fileBuilder = BytesBuilder();

    fileBuilder.add(headerLengthBytes.buffer.asUint8List()); // 8 bytes
    fileBuilder.add(jsonBytes);                              // JSON String
    fileBuilder.add(rawBuffer.takeBytes());                  // Raw F32 Tensor Data

    file.writeAsBytesSync(fileBuilder.takeBytes());

    Logger.log('Export Complete! Saved successfully to $outputPath');
  }
}