
/// Export the layer classes
library;
export 'averagePoolingLayer.dart';
export 'batchNormalizationLayer.dart';
export 'conv2D.dart';
export 'convlstmLayer.dart';
export 'denseLayer.dart';
export 'dropoutLayer.dart';
export 'dualLSTM.dart';
export 'embeddingLayer.dart';
export 'flattenLayer.dart';
export 'globalAveragePoolingLayer.dart';
export 'lstmLayer.dart';
export 'maxPoolingLayer.dart';
export 'multiHeadAttentionLayer.dart';
export 'multiLSTMLayer.dart';
export 'normalizationLayer.dart';
export 'positionalEncodingLayer.dart';
export 'reluLayer.dart';
export 'rnnLayer.dart';
export 'singleHeadAttentionLayer.dart';
export 'transformerEncodingLayer.dart';


import '../../tensor/tensor.dart';

abstract class Layer<I, O> {
  List<Tensor<dynamic>> get parameters;

  Map<String, dynamic> getWeights();
  void setWeights(Map<String, dynamic> weights);

  String get name;

  bool _built = false;

  void build(Tensor<I> input) {
    _built = true;
  }

  Tensor<O> forward(Tensor<I> input);

  Tensor<O> call(Tensor<I> input) {
    if (!_built) {
      build(input);
    }
    return forward(input);
  }
}