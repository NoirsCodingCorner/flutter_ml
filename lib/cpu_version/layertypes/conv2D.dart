import 'dart:math';

import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import '../activationFuncitons/activationFunction.dart';
import 'layer.dart';

/// Applies a 2D convolution over an input [Matrix] tensor.
/// Convolves the input with a set of learnable 2D filters ([kernels]) to produce a 3D feature map ([Tensor3D]),
/// adds channel-wise [biases], and optionally applies an [activation] function.
class Conv2DLayer extends Layer<Matrix, Tensor3D> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'conv2d_layer';

  /// Number of output channels (feature maps) to produce.
  int outChannels;

  /// Spatial height and width of the 2D square convolution kernels.
  int kernelSize;

  /// Padding scheme used during convolution ('valid' or 'same').
  String padding;

  /// Optional activation function applied element-wise to the output 3D tensor.
  ActivationFunction<Tensor3D>? activation;

  /// List of learnable 2D kernel tensors, one per output channel.
  late List<Tensor<Matrix>> kernels;

  /// Learnable bias vector tensor containing one bias per output channel.
  late Tensor<Vector> biases;

  /// Creates a [Conv2DLayer] with [outChannels], [kernelSize], and optional [padding] and [activation].
  Conv2DLayer(
      this.outChannels,
      this.kernelSize, {
        this.padding = 'valid',
        this.activation,
      });

  /// Returns all trainable parameters including each kernel tensor and the bias vector.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    for (int i = 0; i < kernels.length; i = i + 1) {
      params.add(kernels[i]);
    }
    params.add(biases);
    return params;
  }

  /// Allocates and initializes the convolution kernels using standard scaling and sets biases to 0.0.
  @override
  void build(Tensor<Matrix> input) {
    Random random = Random();
    kernels = [];

    double stddev = sqrt(2.0 / (kernelSize * kernelSize));

    for (int i = 0; i < outChannels; i = i + 1) {
      Matrix kernelValues = [];
      for (int r = 0; r < kernelSize; r = r + 1) {
        Vector row = [];
        for (int c = 0; c < kernelSize; c = c + 1) {
          row.add((random.nextDouble() * 2.0 - 1.0) * stddev);
        }
        kernelValues.add(row);
      }
      kernels.add(Tensor<Matrix>(kernelValues));
    }

    Vector biasValues = [];
    for (int i = 0; i < outChannels; i = i + 1) {
      biasValues.add(0.0);
    }
    biases = Tensor<Vector>(biasValues);

    super.build(input);
  }

  /// Executes the 2D convolution forward pass on the CPU, adding biases and applying activation if defined.
  @override
  Tensor<Tensor3D> forward(Tensor<Matrix> input) {
    List<Tensor<Matrix>> featureMaps = [];

    for (int i = 0; i < outChannels; i = i + 1) {
      Tensor<Matrix> featureMap = conv2d(input, kernels[i], padding: padding);
      Tensor<Scalar> bias = Tensor<Scalar>(biases.value[i]);
      Tensor<Matrix> biasedMap = addScalarToMatrix(featureMap, bias);
      featureMaps.add(biasedMap);
    }

    Tensor<Tensor3D> out = stackMatricesTo3D(featureMaps);

    if (activation != null) {
      return activation!.call(out);
    } else {
      return out;
    }
  }

  /// Returns the kernel matrices and bias values as a map.
  @override
  Map<String, dynamic> getWeights() {
    List<Matrix> kernelValues = [];
    for (int i = 0; i < kernels.length; i = i + 1) {
      kernelValues.add(kernels[i].value);
    }

    Map<String, dynamic> weightsMap = {};
    weightsMap['kernels'] = kernelValues;
    weightsMap['biases'] = biases.value;
    return weightsMap;
  }

  /// Sets the kernel matrices and bias values from a provided weights map.
  @override
  void setWeights(Map<String, dynamic> weightsMap) {
    List<dynamic> newKernelValues = weightsMap['kernels'] as List<dynamic>;
    List<dynamic> biasesDynamic = weightsMap['biases'] as List<dynamic>;

    for (int i = 0; i < biasesDynamic.length; i = i + 1) {
      biases.data[i] = biasesDynamic[i] as double;
    }

    for (int i = 0; i < kernels.length; i = i + 1) {
      List<dynamic> kernelDynamic = newKernelValues[i] as List<dynamic>;

      int idx = 0;
      for (int r = 0; r < kernelSize; r = r + 1) {
        List<dynamic> rowDynamic = kernelDynamic[r] as List<dynamic>;
        for (int c = 0; c < kernelSize; c = c + 1) {
          kernels[i].data[idx] = rowDynamic[c] as double;
          idx = idx + 1;
        }
      }
    }
  }
}

/// Computes the Mean Squared Error (MSE) loss between flattened elements of [pred] and [target] tensors.
/// Builds an autograd [Node] attached to the returned [Scalar] [Tensor] to propagate gradients back to [pred].
Tensor<Scalar> flatMse(Tensor<dynamic> pred, Tensor<dynamic> target) {
  double sum = 0.0;
  int length = pred.data.length;

  for (int i = 0; i < length; i = i + 1) {
    double diff = pred.data[i] - target.data[i];
    sum = sum + (diff * diff);
  }
  double mseValue = sum / length;

  Tensor<Scalar> loss = Tensor<Scalar>(mseValue);

  loss.creator = Node(
    [pred, target],
        () {
      double gradMultiplier = 2.0 / length;
      for (int i = 0; i < length; i = i + 1) {
        double diff = pred.data[i] - target.data[i];
        pred.grad[i] = pred.grad[i] + loss.grad[0] * diff * gradMultiplier;
      }
    },
    opName: 'flat_mse',
    cost: length,
  );

  return loss;
}

/*void main() {
  int inputSize = 5;
  int outChannels = 2;
  int kernelSize = 3;

  Matrix inputData = [];
  for (int i = 0; i < inputSize; i = i + 1) {
    Vector row = [];
    for (int j = 0; j < inputSize; j = j + 1) {
      row.add((i + j).toDouble());
    }
    inputData.add(row);
  }
  Tensor<Matrix> input = Tensor<Matrix>(inputData);

  Conv2DLayer conv2d = Conv2DLayer(outChannels, kernelSize, padding: 'valid');
  conv2d.build(input);

  int outputSize = inputSize - kernelSize + 1;
  Tensor3D targetData = [];
  for (int c = 0; c < outChannels; c = c + 1) {
    Matrix m = [];
    for (int i = 0; i < outputSize; i = i + 1) {
      Vector row = [];
      for (int j = 0; j < outputSize; j = j + 1) {
        row.add(1.0);
      }
      m.add(row);
    }
    targetData.add(m);
  }
  Tensor<Tensor3D> target = Tensor<Tensor3D>(targetData);

  SGD optimizer = SGD(conv2d.parameters, learningRate: 0.01);

  for (step = 0; step < 50; step = step + 1) {
    Tensor<Tensor3D> output = conv2d.forward(input);
    Tensor<Scalar> loss = flatMse(output, target);

    print(loss.value);

    loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
  }
}*/