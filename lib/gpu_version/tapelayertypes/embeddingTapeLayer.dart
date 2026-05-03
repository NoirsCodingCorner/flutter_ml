import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class EmbeddingTL extends TapeLayer {
  int vocabularySize;
  int embeddingDimension;

  late GPUTensor<Matrix> embeddings;

  EmbeddingTL(this.vocabularySize, this.embeddingDimension);

  @override
  String get name {
    return 'EmbeddingTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(embeddings);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings = GPUTensor<Matrix>(values);
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Vector> typedInput = input as GPUTensor<Vector>;
    return embeddingLookupGPU(typedInput, embeddings, tape);
  }

  @override
  void free() {
    if (built) {
      embeddings.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.toCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      embeddings.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    embeddings = GPUTensor<Matrix>(wData);
    built = true;
  }
}

class EmbeddingMatrixTL extends TapeLayer {
  int vocabularySize;
  int embeddingDimension;

  late GPUTensor<Matrix> embeddings;

  EmbeddingMatrixTL(this.vocabularySize, this.embeddingDimension);

  @override
  String get name {
    return 'EmbeddingMatrixTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(embeddings);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    Random random = Random();
    List<List<double>> values = <List<double>>[];

    for (int i = 0; i < vocabularySize; i = i + 1) {
      List<double> row = <double>[];
      for (int j = 0; j < embeddingDimension; j = j + 1) {
        row.add((random.nextDouble() * 2.0 - 1.0) * 0.01);
      }
      values.add(row);
    }

    embeddings = GPUTensor<Matrix>(values);
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    return embeddingLookupBatchGPU(typedInput, embeddings, tape);
  }

  @override
  void free() {
    if (built) {
      embeddings.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    embeddings.toCpu();
    wMap['embeddings'] = embeddings.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      embeddings.free();
    }

    List<List<double>> wData = <List<double>>[];
    List<dynamic> rawW = newWeights['embeddings']!;
    for (int i = 0; i < rawW.length; i = i + 1) {
      List<double> row = <double>[];
      List<dynamic> rawRow = rawW[i] as List<dynamic>;
      for (int j = 0; j < rawRow.length; j = j + 1) {
        row.add(rawRow[j] as double);
      }
      wData.add(row);
    }

    embeddings = GPUTensor<Matrix>(wData);
    built = true;
  }
}