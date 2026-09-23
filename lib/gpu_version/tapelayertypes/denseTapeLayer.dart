import 'dart:math';
import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

// Assuming your TapeLayer abstract class is imported here

class DenseTL extends TapeLayer {
  int outputSize;

  late GPUTensor<Matrix> weights;
  late GPUTensor<Vector> bias;

  DenseTL(this.outputSize);

  @override
  String get name {
    return 'DenseLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(bias);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    int inputSize = input.shape[1]; //

    Random rnd = Random();
    List<List<double>> wData = <List<double>>[];

    // Simple Xavier initialization scale
    double scale = sqrt(2.0 / (inputSize + outputSize));

    for (int i = 0; i < inputSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add((rnd.nextDouble() * 2.0 - 1.0) * scale);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    for (int i = 0; i < outputSize; i = i + 1) {
      bData.add(0.0);
    }

    weights = GPUTensor<Matrix>(wData); //
    bias = GPUTensor<Vector>(bData); //

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    // Capture the intermediate matrix multiplication result in the trash list.
    // This ensures its VRAM can be released after the tape has finished executing.
    GPUTensor<Matrix> mulResult = matMulGPU(typedInput, weights, tape); //
    intermediates.add(mulResult);

    GPUTensor<Matrix> out = addMatrixAndVectorGPU(mulResult, bias, tape);

    return out;
  }

  @override
  void free() {
    if (built) {
      weights.free(); //
      bias.free(); //
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    // Pull from VRAM to CPU before extracting the values
    weights.toCpu();
    bias.toCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      weights.free(); //
      bias.free(); //
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    // Instantiating the GPUTensor automatically pushes this data to VRAM
    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);
    built = true;
  }
}

class DenseReluTL extends TapeLayer {
  int outputSize;

  late GPUTensor<Matrix> weights;
  late GPUTensor<Vector> bias;

  DenseReluTL(this.outputSize);

  @override
  String get name {
    return 'DenseReluLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(bias);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    int inputSize = input.shape[1];

    Random rnd = Random();
    List<List<double>> wData = <List<double>>[];

    double scale = sqrt(2.0 / (inputSize + outputSize));

    for (int i = 0; i < inputSize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add((rnd.nextDouble() * 2.0 - 1.0) * scale);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    for (int i = 0; i < outputSize; i = i + 1) {
      bData.add(0.0);
    }

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    GPUTensor<Matrix> mulResult = matMulGPU(typedInput, weights, tape);
    intermediates.add(mulResult);

    GPUTensor<Matrix> out = addMatrixAndVectorGPU(mulResult, bias, tape);
    intermediates.add(out); // <--- FEHLTE! Extrem wichtig für Zero-Grad!

    return out;
  }

  @override
  void free() {
    if (built) {
      weights.free();
      bias.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    weights.toCpu();
    bias.toCpu();

    wMap['weights'] = weights.value;
    wMap['bias'] = bias.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      weights.free();
      bias.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['weights']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    List<double> bData = <double>[];
    List<dynamic> rawB = newWeights['bias']!;
    for (int i = 0; i < rawB.length; i = i + 1) {
      bData.add(rawB[i] as double);
    }

    weights = GPUTensor<Matrix>(wData);
    bias = GPUTensor<Vector>(bData);
    built = true;
  }
}
