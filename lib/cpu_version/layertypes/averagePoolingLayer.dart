import '../../tensor/tensor.dart';
import '../../tensor/tensor_math_cpu.dart';
import '../../tensor/type_Aliases.dart';
import 'layer.dart';

class AveragePooling2DLayer extends Layer<Matrix, Matrix> {
  @override
  String name = 'average_pooling_2d';
  int poolSize;
  int stride;

  late int inputHeight;
  late int inputWidth;

  AveragePooling2DLayer({this.poolSize = 2, this.stride = 2});

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  void build(Tensor<Matrix> input) {
    inputHeight = input.shape[0];
    inputWidth = input.shape[1];
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<Matrix> input) {
    return avgPool2d(input, poolSize, stride);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}

class GlobalAveragePoolingLayer extends Layer<Matrix, Vector> {
  @override
  String name = 'global_avg_pool';

  @override
  List<Tensor<dynamic>> get parameters {
    List<Tensor<dynamic>> params = [];
    return params;
  }

  @override
  void build(Tensor<Matrix> input) {
    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<Matrix> input) {
    return globalAveragePooling(input);
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> emptyMap = {};
    return emptyMap;
  }

  @override
  void setWeights(Map<String, dynamic> weights) {}
}
