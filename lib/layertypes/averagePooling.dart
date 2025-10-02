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

