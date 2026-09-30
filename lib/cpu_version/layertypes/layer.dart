/// Holds and exports all the different layer types that can be used for CPU model building.
library;

export 'average_pooling_layer.dart';
export 'batch_normalization_layer.dart';
export 'conv_2d.dart';
export 'convlstm_layer.dart';
export 'dense_layer.dart';
export 'dropout_layer.dart';
export 'dual_lstm.dart';
export 'embedding_layer.dart';
export 'flatten_layer.dart';
export 'global_average_pooling_layer.dart';
export 'lstm_layer.dart';
export 'max_pooling_layer.dart';
export 'multi_head_attention_layer.dart';
export 'multi_lstm_layer.dart';
export 'normalization_layer.dart';
export 'positional_encoding_layer.dart';
export 'relu_layer.dart';
export 'rnn_layer.dart';
export 'single_head_attention_layer.dart';
export 'transformer_encoding_layer.dart';

import '../../tensor/tensor.dart';

/// Layer is built to be an additional abstraction that allows for complex mathematical bundling on the CPU.
/// Operations like Dense Neural Networks are supported as a single Layer and can be used to build models reliably.
/// Each Layer has the following individually assigned values:
///
/// -[name]: String name of the layer type to allow for additional debugging features.
///
/// -[_built]: Bool flag to register whether a Layer has already been built and initialized.
///
/// -[parameters]: The list of [Tensor] instances of this Layer which are to be adjusted via a learning function such as Adam or SGD.
///
/// The Layer supports the following functions:
///
/// -[build]: Responsible for allocating and initializing weights based on the provided [Tensor] input shape.
///
/// -[forward]: Computes the forward pass operations on the provided input tensor.
///
/// -[call]: Helper function that, if not overwritten, checks whether [build] has already been called; if not, it calls [build] and only then [forward].
///
/// -[getWeights]: Function to retrieve the weights for storing purposes.
///
/// -[setWeights]: Function to set the weights to the provided values.
abstract class Layer<I, O> {
  /// The list of [Tensor] instances of this Layer which are to be adjusted via a learning function such as Adam or SGD. Those are specifically exposed to be adjusted
  /// via an [Optimizer].
  List<Tensor<dynamic>> get parameters;

  /// Function to retrieve the weights for storing purposes.
  Map<String, dynamic> getWeights();

  /// Function to set the weights to the provided values.
  void setWeights(Map<String, dynamic> weights);

  /// The assigned name of the given Layer architecture. When building custom architectures, it is recommended to use a meaningful name.
  String get name;

  /// Bool flag to register whether a Layer has already been built.
  bool _built = false;

  /// Allocates and initializes weights/biases based on the incoming input shape. The [Tensor] [input] will be used for the setup and therefore
  /// should match the exact [Tensor] shape used in [forward].
  void build(Tensor<I> input) {
    _built = true;
  }

  /// Executes this layer's forward operations on the CPU and returns the resulting [Tensor].
  /// The input tensor should match the tensor provided in the [build] method.
  Tensor<O> forward(Tensor<I> input);

  /// Helper function that, if not overwritten, checks whether [build] has already been called; if not, it calls [build] and only then [forward].
  Tensor<O> call(Tensor<I> input) {
    if (!_built) {
      build(input);
    }
    return forward(input);
  }
}