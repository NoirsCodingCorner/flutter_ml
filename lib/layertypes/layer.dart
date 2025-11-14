import '../autogradEngine/tensor.dart';

export 'averagePooling.dart';
export 'batchNormalizationLayer.dart';
export 'conv2d.dart';
export 'convlstmLayer.dart';
export 'denseLayer.dart';
export 'dropout.dart';
export 'duallstm.dart';
export 'flattenLayer.dart';
export 'lstmLayer.dart';
export 'maxPooling.dart';
export 'reluLayer.dart';
export 'rnnLayer.dart';
export 'singleHeadAttentionLayer.dart';
export 'trendmodelLayer.dart';

abstract class Layer {
  List<Tensor> get parameters;

  Map<String, dynamic> getWeights();
  void setWeights(Map<String, dynamic> weights);

  String get name;

  bool _built = false;

  void build(Tensor<dynamic> input) {
    _built = true;
  }

  Tensor<dynamic> forward(Tensor<dynamic> input);

  Tensor<dynamic> call(Tensor<dynamic> input) {
    if (!_built) {
      build(input);
    }
    return forward(input);
  }
}