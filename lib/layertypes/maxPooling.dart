import '../autogradEngine/tensor.dart';
import 'averagePooling.dart';
import 'layer.dart';

class MaxPooling2DLayer extends Layer {
  @override
  String name = 'max_pooling_2d';
  int poolSize;
  int stride;

  late int inputHeight, inputWidth;
  late List<List<List<int>>> maxIndices;

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

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}

class MaxPooling1DLayer extends Layer {
  @override
  String name = 'max_pooling_1d';
  int poolSize;
  int stride;

  late int inputSize;
  late List<int> maxIndices;

  MaxPooling1DLayer({this.poolSize = 2, this.stride = 2});

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    Vector inputValue = input.value as Vector;
    inputSize = inputValue.length;
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Vector inputValue = (input as Tensor<Vector>).value;
    int outputSize = (inputSize - poolSize) ~/ stride + 1;

    Vector outputValue = [];
    maxIndices = [];

    for (int i = 0; i < outputSize; i++) {
      double maxVal = -double.infinity;
      int max_i = -1;

      for (int p = 0; p < poolSize; p++) {
        int currentIndex = i * stride + p;
        if (inputValue[currentIndex] > maxVal) {
          maxVal = inputValue[currentIndex];
          max_i = currentIndex;
        }
      }
      outputValue.add(maxVal);
      maxIndices.add(max_i);
    }

    Tensor<Vector> out = Tensor<Vector>(outputValue);
    out.creator = Node([input], () {
      for (int i = 0; i < outputSize; i++) {
        int max_i = maxIndices[i];
        input.grad[max_i] += out.grad[i];
      }
    }, opName: 'max_pool_1d');

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}