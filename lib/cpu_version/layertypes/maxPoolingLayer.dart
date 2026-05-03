import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class MaxPooling2DLayer extends Layer<Matrix, Matrix> {
  @override
  String name = 'max_pooling_2d';
  int poolSize;
  int stride;

  late int inputHeight;
  late int inputWidth;
  late List<int> maxIndicesFlat;

  MaxPooling2DLayer({this.poolSize = 2, this.stride = 2});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  void build(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    inputHeight = inputMatrix.length;
    if (inputHeight > 0) {
      inputWidth = inputMatrix[0].length;
    } else {
      inputWidth = 0;
    }
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    Matrix inputMatrix = input.value;
    int outputHeight = (inputHeight - poolSize) ~/ stride + 1;
    int outputWidth = (inputWidth - poolSize) ~/ stride + 1;

    Matrix outputValue = [];
    maxIndicesFlat = [];

    for (int y = 0; y < outputHeight; y = y + 1) {
      Vector row = [];
      for (int x = 0; x < outputWidth; x = x + 1) {
        double maxVal = -double.infinity;
        int maxFlatIndex = -1;

        for (int py = 0; py < poolSize; py = py + 1) {
          for (int px = 0; px < poolSize; px = px + 1) {
            int current_y = y * stride + py;
            int current_x = x * stride + px;
            if (inputMatrix[current_y][current_x] > maxVal) {
              maxVal = inputMatrix[current_y][current_x];
              // Calculate the 1D flat index for the gradient array
              maxFlatIndex = current_y * inputWidth + current_x;
            }
          }
        }
        row.add(maxVal);
        maxIndicesFlat.add(maxFlatIndex);
      }
      outputValue.add(row);
    }

    Tensor<Matrix> out = Tensor<Matrix>(outputValue);
    out.creator = Node(
      [input],
          () {
        int idx = 0;
        for (int y = 0; y < outputHeight; y = y + 1) {
          for (int x = 0; x < outputWidth; x = x + 1) {
            int inFlatIdx = maxIndicesFlat[idx];
            int outFlatIdx = y * outputWidth + x;

            input.grad[inFlatIdx] = input.grad[inFlatIdx] + out.grad[outFlatIdx];
            idx = idx + 1;
          }
        }
      },
      opName: 'max_pool_2d',
      cost: outputHeight * outputWidth * poolSize * poolSize,
    );

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}

class MaxPooling1DLayer extends Layer<Vector, Vector> {
  @override
  String name = 'max_pooling_1d';
  int poolSize;
  int stride;

  late int inputSize;
  late List<int> maxIndices;

  MaxPooling1DLayer({this.poolSize = 2, this.stride = 2});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  void build(Tensor<Vector> input) {
    Vector inputValue = input.value;
    inputSize = inputValue.length;
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    Vector inputValue = input.value;
    int outputSize = (inputSize - poolSize) ~/ stride + 1;

    Vector outputValue = [];
    maxIndices = [];

    for (int i = 0; i < outputSize; i = i + 1) {
      double maxVal = -double.infinity;
      int max_i = -1;

      for (int p = 0; p < poolSize; p = p + 1) {
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
    out.creator = Node(
      [input],
          () {
        for (int i = 0; i < outputSize; i = i + 1) {
          int max_i = maxIndices[i];
          input.grad[max_i] = input.grad[max_i] + out.grad[i];
        }
      },
      opName: 'max_pool_1d',
      cost: outputSize * poolSize,
    );

    return out;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}