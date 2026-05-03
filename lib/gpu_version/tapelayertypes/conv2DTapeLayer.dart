import 'dart:math';

import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

class Conv2DTL extends TapeLayer {
  int outChannels;
  int kernelSize;
  String padding;

  late int inChannels;
  late GPUTensor<Tensor3D> weights;
  late GPUTensor<Vector> biases;

  Conv2DTL(this.outChannels, this.kernelSize, {this.padding = 'valid'});

  @override
  String get name {
    return 'Conv2DTapeLayer';
  }

  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(biases);
    }
    return params;
  }

  @override
  void build(GPUTensor<dynamic> input) {
    inChannels = input.shape.length == 2 ? 1 : input.shape[0];
    Random rnd = Random();

    double stddev = sqrt(2.0 / (kernelSize * kernelSize * inChannels));

    List<List<List<double>>> weightData = <List<List<double>>>[];
    for (int oc = 0; oc < outChannels; oc = oc + 1) {
      List<List<double>> cInList = <List<double>>[];
      for (int ic = 0; ic < inChannels; ic = ic + 1) {
        List<double> hwList = <double>[];
        for (int h = 0; h < kernelSize * kernelSize; h = h + 1) {
          hwList.add((rnd.nextDouble() * 2.0 - 1.0) * stddev);
        }
        cInList.add(hwList);
      }
      weightData.add(cInList);
    }

    List<double> biasData = <double>[];
    for (int i = 0; i < outChannels; i = i + 1) {
      biasData.add(0.0);
    }

    weights = GPUTensor<Tensor3D>(weightData);
    biases = GPUTensor<Vector>(biasData);

    built = true;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Tensor3D> out = conv2dMultiChannelGPU(
        input,
        weights,
        biases,
        kernelSize,
        kernelSize,
        tape,
        padding: padding
    );

    return out;
  }

  @override
  void free() {
    if (built) {
      weights.free();
      biases.free();
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (built == false) {
      return wMap;
    }

    weights.toCpu();
    biases.toCpu();

    wMap['weights'] = weights.value;
    wMap['biases'] = biases.value;

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) {
      weights.free();
      biases.free();
    }

    List<dynamic> rawWeights = newWeights['weights']!;
    List<dynamic> rawBiases = newWeights['biases']!;

    List<List<List<double>>> weightData = <List<List<double>>>[];
    for (int oc = 0; oc < outChannels; oc = oc + 1) {
      List<List<double>> cInList = <List<double>>[];
      List<dynamic> rawInMap = rawWeights[oc] as List<dynamic>;

      for (int ic = 0; ic < inChannels; ic = ic + 1) {
        List<double> hwList = <double>[];
        List<dynamic> rawHwMap = rawInMap[ic] as List<dynamic>;

        for (int h = 0; h < kernelSize * kernelSize; h = h + 1) {
          hwList.add(rawHwMap[h] as double);
        }
        cInList.add(hwList);
      }
      weightData.add(cInList);
    }

    List<double> biasData = <double>[];
    for (int i = 0; i < outChannels; i = i + 1) {
      biasData.add(rawBiases[i] as double);
    }

    weights = GPUTensor<Tensor3D>(weightData);
    biases = GPUTensor<Vector>(biasData);

    built = true;
  }
}