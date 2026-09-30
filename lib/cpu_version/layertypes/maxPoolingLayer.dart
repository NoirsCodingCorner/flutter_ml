import '../../tensor/tensor.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

/// Applies a 2D max pooling operation over an input [Matrix] tensor.
/// Downsamples the spatial representation by extracting the maximum value within pooling windows defined by [poolSize] and [stride].
/// Tracks the flat 1D input indices of maximum values to route backpropagating gradients directly to those elements via an autograd [Node].
class MaxPooling2DLayer extends Layer<Matrix, Matrix> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'max_pooling_2d';

  /// Spatial height and width of the square pooling window.
  int poolSize;

  /// The step size of the pooling window across the input matrix.
  int stride;

  /// Cached height dimension of the input matrix.
  late int inputHeight;

  /// Cached width dimension of the input matrix.
  late int inputWidth;

  /// Flat indices storing the positions of the maximum elements in the input matrix for backward gradient routing.
  late List<int> maxIndicesFlat;

  /// Creates a [MaxPooling2DLayer] with the given [poolSize] and [stride].
  MaxPooling2DLayer({this.poolSize = 2, this.stride = 2});

  /// Returns the trainable parameters. Since max pooling contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Caches the spatial dimensions of the input matrix and registers the layer as built.
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

  /// Executes 2D max pooling on the CPU, storing winning indices and building an autograd [Node] for gradient backpropagation.
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
            int currentY = y * stride + py;
            int currentX = x * stride + px;
            if (inputMatrix[currentY][currentX] > maxVal) {
              maxVal = inputMatrix[currentY][currentX];
              // Calculate the 1D flat index for the gradient array
              maxFlatIndex = currentY * inputWidth + currentX;
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

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}

/// Applies a 1D max pooling operation over an input [Vector] tensor.
/// Downsamples the 1D representation by selecting the maximum value within sliding windows defined by [poolSize] and [stride].
/// Tracks input indices of maximum values to route backpropagating gradients directly to those elements via an autograd [Node].
class MaxPooling1DLayer extends Layer<Vector, Vector> {
  /// The assigned name of this layer architecture for debugging and inspection.
  @override
  String name = 'max_pooling_1d';

  /// The size of the sliding 1D pooling window.
  int poolSize;

  /// The step size of the pooling window across the input vector.
  int stride;

  /// Cached element count of the input vector.
  late int inputSize;

  /// Indices storing the positions of the maximum elements in the input vector for backward gradient routing.
  late List<int> maxIndices;

  /// Creates a [MaxPooling1DLayer] with the given [poolSize] and [stride].
  MaxPooling1DLayer({this.poolSize = 2, this.stride = 2});

  /// Returns the trainable parameters. Since max pooling contains no trainable parameters, an empty list is returned.
  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  /// Caches the element count of the input vector and registers the layer as built.
  @override
  void build(Tensor<Vector> input) {
    Vector inputValue = input.value;
    inputSize = inputValue.length;
    super.build(input);
  }

  /// Executes 1D max pooling on the CPU, storing winning indices and building an autograd [Node] for gradient backpropagation.
  @override
  Tensor<Vector> forward(Tensor<Vector> input) {
    Vector inputValue = input.value;
    int outputSize = (inputSize - poolSize) ~/ stride + 1;

    Vector outputValue = [];
    maxIndices = [];

    for (int i = 0; i < outputSize; i = i + 1) {
      double maxVal = -double.infinity;
      int maxI = -1;

      for (int p = 0; p < poolSize; p = p + 1) {
        int currentIndex = i * stride + p;
        if (inputValue[currentIndex] > maxVal) {
          maxVal = inputValue[currentIndex];
          maxI = currentIndex;
        }
      }
      outputValue.add(maxVal);
      maxIndices.add(maxI);
    }

    Tensor<Vector> out = Tensor<Vector>(outputValue);
    out.creator = Node(
      [input],
          () {
        for (int i = 0; i < outputSize; i = i + 1) {
          int maxI = maxIndices[i];
          input.grad[maxI] = input.grad[maxI] + out.grad[i];
        }
      },
      opName: 'max_pool_1d',
      cost: outputSize * poolSize,
    );

    return out;
  }

  /// Returns an empty map as this layer contains no trainable weights.
  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  /// No-op as this layer contains no weights to set.
  @override
  void setWeights(Map<String, dynamic> weights) {}
}