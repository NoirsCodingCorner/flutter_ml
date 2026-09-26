import 'dart:math';

import '../../tensor/tensor_gpu.dart';
import '../../tensor/tensor_math_gpu.dart';
import '../../tensor/type_Aliases.dart';
import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// Applies a 2D convolution over a multi-channel input using learnable spatial filters.
/// Works for Matrix and Tensor3D inputs. On Matrix input it is assumed to contain 1 channel.
class Conv2DTL extends TapeLayer<dynamic,Tensor3D> {
  String get name => 'Conv2DTapeLayer';

  int outChannels;
  int kernelSize;
  String padding;

  /// Number of input channels.
  late int inChannels;
  /// Learnable filter weights for the convolution.
  late GPUTensor<Tensor3D> weights;
  /// Learnable shifting vector (similar to bias) applied to each output channel.
  late GPUTensor<Vector> biases;

  /// Persistent Cache for Static Unrolling
  int cacheBatchSize = -1;
  GPUTensor<Tensor3D>? cachedOut;

  /// Requires the number of desired filters [outChannels] and the spatial size of the square kernel [kernelSize].
  /// Takes an optional [padding] parameter ('valid' or 'same') to dictate output dimensions.
  Conv2DTL(this.outChannels, this.kernelSize, {this.padding = 'valid'});

  /// Returns the trainable parameters [weights] and [biases].
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.add(weights);
      params.add(biases);
    }
    return params;
  }

  /// Allocates VRAM for [weights] and [biases] dynamically determining [inChannels] based on the [input] shape.
  /// Weights are initialized using He normal initialization for stabilization.
  /// Works for Matrix and Tensor3D inputs. On Matrix Input it is assumed to contain 1 channel.
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

  /// Writes the [conv2dMultiChannelGPU] operation to the provided [tape] and returns the [GPUTensor] where the result will be stored.
  /// Persistently caches the output tensor to prevent VRAM leaks and infinite accumulation.
  @override
  GPUTensor<Tensor3D> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int currentBatchSize = input.shape[0];

    // Safely free the old cache if the batch size changes
    if (cacheBatchSize != currentBatchSize) {
      if (cachedOut != null) {
        cachedOut!.free();
      }
      cachedOut = null;
      cacheBatchSize = currentBatchSize;
    }

    cachedOut = conv2dMultiChannelGPU(
        input,
        weights,
        biases,
        kernelSize,
        kernelSize,
        tape,
        padding: padding,
        outTensor: cachedOut
    );

    return cachedOut!;
  }

  /// Clears the gradients of the statically cached output tensor.
  void zeroStates(CommandBuffer tape) {
    if (cachedOut != null) {
      cachedOut!.zeroGrad(tape);
    }
  }

  /// Frees VRAM for [weights], [biases], and the cached output tensor.
  @override
  void free() {
    if (built) {
      weights.free();
      biases.free();
    }
    if (cachedOut != null) {
      cachedOut!.free();
    }
  }

  /// Returns [weights] and [biases] as a map.
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

  /// Sets [weights] and [biases] using the provided map.
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