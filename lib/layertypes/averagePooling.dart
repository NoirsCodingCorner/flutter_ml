import '../autogradEngine/tensor.dart';
import 'layer.dart';

/// A 2D average pooling layer.
///
/// This layer downsamples an input feature map by taking the average value
/// over a specified window (`poolSize`). It provides a smoother form of
/// downsampling compared to `MaxPooling2DLayer`.
///
/// - **Input:** A `Tensor<Matrix>` representing a single feature map of shape
///   `[height, width]`.
/// - **Output:** A `Tensor<Matrix>` representing the downsampled feature map.
///
/// ### Example
/// ```dart
/// SNetwork model = SNetwork([
///   Conv2DLayer(16, 3, activation: ReLU()),
///   AveragePooling2DLayer(poolSize: 2, stride: 2),
/// ]);
/// ```
class AveragePooling2DLayer extends Layer {
  @override
  String name = 'average_pooling_2d';
  int poolSize;
  int stride;

  late int inputHeight, inputWidth;

  AveragePooling2DLayer({this.poolSize = 2, this.stride = 2});

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    inputHeight = inputMatrix.length;
    inputWidth = inputMatrix[0].length;
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Matrix inputMatrix = (input as Tensor<Matrix>).value;
    int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
    int outputWidth = (inputWidth - poolSize) ~/ stride + 1;
    int poolArea = poolSize * poolSize;

    Matrix outputValue = [];
    for (int y = 0; y < outputHeight; y++) {
      Vector row = [];
      for (int x = 0; x < outputWidth; x++) {
        double sum = 0;
        for (int py = 0; py < poolSize; py++) {
          for (int px = 0; px < poolSize; px++) {
            sum += inputMatrix[y * stride + py][x * stride + px];
          }
        }
        row.add(sum / poolArea);
      }
      outputValue.add(row);
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputValue);
    out.creator = Node([input], () {
      for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
          double distributed_grad = out.grad[y][x] / poolArea;
          for (int py = 0; py < poolSize; py++) {
            for (int px = 0; px < poolSize; px++) {
              input.grad[y * stride + py][x * stride + px] += distributed_grad;
            }
          }
        }
      }
    }, opName: 'avg_pool_2d');

    return out;
  }
}

// Assumes Layer and Vector types are defined.

/// A layer that reduces a sequence Matrix [seq_len, dModel] to a single
/// Vector [dModel] by averaging across the sequence dimension.
class GlobalAveragePoolingLayer extends Layer {
  @override
  String name = 'global_avg_pool';

  @override
  List<Tensor> get parameters => <Tensor>[];

  @override
  // No build needed as it is purely a mathematical operation.
  void build(Tensor<dynamic> input) {
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    return globalAveragePooling(input as Tensor<Matrix>);
  }
}

// Assumes Node and Tensor types are defined.

/// Computes the average of a Tensor<Matrix> across the sequence dimension (rows).
/// Input: [sequence_length, dModel]. Output: [dModel].
Tensor<Vector> globalAveragePooling(Tensor<Matrix> input) {
  Matrix inputMatrix = input.value as Matrix;
  int sequenceLength = inputMatrix.length;
  int dModel = inputMatrix[0].length;

  Vector averagedVector = List<double>.filled(dModel, 0.0);

  // Summing elements across the sequence dimension (rows)
  for (int r = 0; r < sequenceLength; r++) {
    for (int c = 0; c < dModel; c++) {
      averagedVector[c] += inputMatrix[r][c];
    }
  }

  // Averaging
  for (int c = 0; c < dModel; c++) {
    averagedVector[c] /= sequenceLength.toDouble();
  }

  Tensor<Vector> out = Tensor<Vector>(averagedVector);

  // Backward pass: Gradient is distributed equally back to all rows of the input matrix
  out.creator = Node(<Tensor>[input], () {
    for (int r = 0; r < sequenceLength; r++) {
      for (int c = 0; c < dModel; c++) {
        input.grad[r][c] += out.grad[c] / sequenceLength.toDouble();
      }
    }
  }, opName: 'global_avg_pool', cost: sequenceLength * dModel);

  return out;
}