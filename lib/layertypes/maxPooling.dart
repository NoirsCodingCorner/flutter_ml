import '../autogradEngine/tensor.dart';
import 'averagePooling.dart';
import 'layer.dart';

/// A 2D max pooling layer.
///
/// This layer downsamples an input feature map by taking the maximum value
/// over a specified window (`poolSize`). This helps to make the representation
/// more compact and more robust to the precise location of features.
///
/// It is a standard component in most Convolutional Neural Networks, typically
/// applied after a `Conv2DLayer`.
///
/// - **Input:** A `Tensor<Matrix>` representing a single feature map of shape
///   `[height, width]`.
/// - **Output:** A `Tensor<Matrix>` representing the downsampled feature map.
///
/// ### Example
/// ```dart
/// SNetwork model = SNetwork([
///   Conv2DLayer(16, 3, activation: ReLU()),
///   MaxPooling2DLayer(poolSize: 2, stride: 2),
/// ]);
/// ```
class MaxPooling2DLayer extends Layer {
  @override
  String name = 'max_pooling_2d';
  int poolSize;
  int stride;

  late int inputHeight, inputWidth;
  late List<List<List<int>>> maxIndices; // To store [y, x] of max values for backprop

  MaxPooling2DLayer({this.poolSize = 2, this.stride = 2});

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

    Matrix outputValue = [];
    maxIndices = [];

    for (int y = 0; y < outputHeight; y++) {
      Vector row = [];
      List<List<int>> indexRow = [];
      for (int x = 0; x < outputWidth; x++) {
        double maxVal = -double.infinity;
        int max_y = -1, max_x = -1;

        for (int py = 0; py < poolSize; py++) {
          for (int px = 0; px < poolSize; px++) {
            int current_y = y * stride + py;
            int current_x = x * stride + px;
            if (inputMatrix[current_y][current_x] > maxVal) {
              maxVal = inputMatrix[current_y][current_x];
              max_y = current_y;
              max_x = current_x;
            }
          }
        }
        row.add(maxVal);
        indexRow.add([max_y, max_x]);
      }
      outputValue.add(row);
      maxIndices.add(indexRow);
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputValue);
    out.creator = Node([input], () {
      for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
          int max_y = maxIndices[y][x][0];
          int max_x = maxIndices[y][x][1];
          input.grad[max_y][max_x] += out.grad[y][x];
        }
      }
    }, opName: 'max_pool_2d');

    return out;
  }
}
