import 'dart:math';

import '../activationFunctions/activation_funciton.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';
import '../optimizers/optimizers.dart';
import 'layer.dart';

class Conv2DLayer extends Layer {
  @override
  String name = 'conv2d_layer';
  int outChannels;
  int kernelSize;
  String padding;
  ActivationFunction? activation;

  late List<Tensor<Matrix>> kernels;
  late Tensor<Vector> biases;

  Conv2DLayer(
      this.outChannels,
      this.kernelSize, {
        this.padding = 'valid',
        this.activation,
      });

  @override
  List<Tensor> get parameters => [...kernels, biases];

  @override
  void build(Tensor<dynamic> input) {
    Random random = Random();
    kernels = [];

    for (int i = 0; i < outChannels; i++) {
      Matrix kernelValues = [];
      double stddev = sqrt(2.0 / (kernelSize * kernelSize));
      for (int r = 0; r < kernelSize; r++) {
        Vector row = [];
        for (int c = 0; c < kernelSize; c++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        kernelValues.add(row);
      }
      kernels.add(Tensor<Matrix>(kernelValues));
    }

    biases = Tensor<Vector>(List<double>.filled(outChannels, 0.0));
    super.build(input);
  }

  @override
  Tensor<Tensor3D> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;
    Tensor3D outputChannels = [];

    for (int i = 0; i < outChannels; i++) {
      Tensor<Matrix> featureMap = conv2d(inputMatrix, kernels[i], padding: padding);
      Tensor<Scalar> bias = Tensor<Scalar>(biases.value[i]);
      Tensor<Matrix> biasedMap = addScalarToMatrix(featureMap, bias);
      outputChannels.add(biasedMap.value);
    }

    Tensor<Tensor3D> out = Tensor<Tensor3D>(outputChannels);

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    List<Matrix> kernelValues = [];
    for (Tensor<Matrix> kernel in kernels) {
      kernelValues.add(kernel.value);
    }

    return {
      'kernels': kernelValues,
      'biases': biases.value,
    };
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> newKernelValues = weightsMap['kernels'] as List<dynamic>;
    Vector newBiases = (weightsMap['biases'] as List).map((dynamic e) => e as double).toList();

    for (int i = 0; i < kernels.length; i++) {
      List<dynamic> kernelDynamic = newKernelValues[i] as List<dynamic>;
      Matrix newKernel = kernelDynamic.map((dynamic row) {
        return (row as List<dynamic>).map((dynamic val) => val as double).toList();
      }).toList();

      for (int r = 0; r < kernelSize; r++) {
        for (int c = 0; c < kernelSize; c++) {
          kernels[i].value[r][c] = newKernel[r][c];
        }
      }
    }

    for (int i = 0; i < biases.value.length; i++) {
      biases.value[i] = newBiases[i];
    }
  }
}