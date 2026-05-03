import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class PositionalEncodingTL extends TapeLayer {
  int maxLength;
  int dModel;

  late GPUTensor<Matrix> encodingMatrixTransposed;

  PositionalEncodingTL(this.maxLength, this.dModel);

  @override
  String get name {
    return 'PositionalEncodingTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    return <GPUTensor>[];
  }

  @override
  void build(GPUTensor<dynamic> input) {
    List<List<double>> peValuesTransposed = <List<double>>[];

    for (int i = 0; i < dModel; i = i + 1) {
      List<double> row = <double>[];
      for (int pos = 0; pos < maxLength; pos = pos + 1) {
        double angle = pos / pow(10000, (2 * i) / dModel);
        if (i % 2 == 0) {
          row.add(sin(angle));
        } else {
          row.add(cos(angle));
        }
      }
      peValuesTransposed.add(row);
    }

    encodingMatrixTransposed = GPUTensor<Matrix>(peValuesTransposed);
    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;
    int sequenceLength = typedInput.shape[0];

    GPUTensor<Matrix> slicedTransposed = sliceColumnGPU(encodingMatrixTransposed, 0, sequenceLength, tape);
    intermediates.add(slicedTransposed);

    GPUTensor<Matrix> positionalTensor = transposeGPU(slicedTransposed, tape);
    intermediates.add(positionalTensor);

    GPUTensor<Matrix> out = addMatrixGPU(typedInput, positionalTensor, tape);

    // ⚡ FIXED: Register final PE output
    intermediates.add(out);

    return out;
  }

  @override
  void free() {
    if (built) {
      encodingMatrixTransposed.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}