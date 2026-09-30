/// Holds and exports all the different layer types that can be used for model building
library;

export 'average_pooling_tape_layer.dart';
export 'batch_normalization_tape_layer.dart';
export 'conv_2d_tape_layer.dart';
export 'conv_lstm_tape_layer.dart';
export 'dense_tape_layer.dart';
export 'dropout_tape_layer.dart';
export 'dual_lstm_tape_layer.dart';
export 'embedding_tape_layer.dart';
export 'flatten_tape_layer.dart';
export 'gelu_tape_layer.dart';
export 'global_average_pooling_tape_layer.dart';
export 'layer_normalization_tape_layer.dart';
export 'lstm_tape_layer.dart';
export 'max_pooling_tape_layer.dart';
export 'multi_head_attention_tape_layer.dart';
export 'positional_encoding_tape_layer.dart';
export 'relu_tape_layer.dart';
export 'rnn_tape_layer.dart';
export 'sigmoid_matrix_tape_layer.dart';
export 'single_head_attention_tape_layer.dart';
export 'transformer_encoder_block_tape_layer.dart';

import '/cpu_version/cpu_version.dart';

import '/tensor/tensor_gpu.dart';
import '../ffi/command_buffer.dart';

/// TapeLayer is built to be an additional abstraction that allows for complex mathematical bundling.
/// Operations like Dense Neural Networks are supported as a single TapeLayer and can be used to build models reliably.
/// Each TapeLayer has the following individually assigned values:
///
/// -[name]: String name of the layer type to allow for additional debugging features.
///
/// -[built]: Bool flag to register whether a TapeLayer has already been built, or rather whether the required VRAM has been allocated at least once.
///
/// -[parameters]: The list of GPUTensors of this TapeLayer which are to be adjusted via a learning function such as Adam or SGD.
///
/// The TapeLayer supports the following functions:
///
/// -[built]: Responsible for allocating and reserving VRAM on the selected device based on the provided [GPUTensor] input shape.
///
/// -[forward]: Appends the given forward instructions to the provided tape.
///
/// -[call]: Helper function that, if not overwritten, checks whether [build] has already been called; if not, it calls [build] and only then [forward].
///
/// -[free]: Function to free the allocated [parameters] VRAM.
abstract class TapeLayer<InputType, OutputType> {
  /// The assigned name of the given TapeLayer architecture. When building custom architectures, it is recommended to use a meaningful name.
  String get name;

  /// The list of GPUTensors of this TapeLayer which are to be adjusted via a learning function such as Adam or SGD. Those are specifically exposed to be adjusted
  /// via an [Optimizer].
  List<GPUTensor> get parameters;

  /// Bool flag to register whether a TapeLayer has already been built, or rather whether the required VRAM has been allocated at least once.
  bool built = false;

  /// Allocates VRAM for weights/biases based on the incoming input shape. The GPUTensor [input] will be used for the construction of the [CommandBuffer] and therefore
  /// should match the exact GPUTensor used in [forward]. All operations of a TapeLayer are in place, so this shape must not change during training or inference.
  void build(GPUTensor<InputType> input) {
    built = true;
  }

  /// Appends this layer's forward operations to the [tape]. The intermediates list captures transient tensors to prevent VRAM creep on complex architectures.
  /// The input tensor should match the tensor provided in the [build] method.
  GPUTensor<OutputType> forward(GPUTensor<InputType> input, CommandBuffer tape,
      List<GPUTensor> intermediates);

  /// Helper function that, if not overwritten, checks whether [build] has already been called; if not, it calls [build] and only then [forward].
  GPUTensor<OutputType> call(GPUTensor<InputType> input, CommandBuffer tape,
      List<GPUTensor> intermediates) {
    if (built == false) {
      build(input);
    }
    return forward(input, tape, intermediates);
  }

  void zeroStates(CommandBuffer tape);

  /// Frees all parameters this model contains. After this is called, the TapeLayer should not be used further.
  void free() {
    List<GPUTensor> params = parameters;
    for (int i = 0; i < params.length; i = i + 1) {
      params[i].free(); //
    }
  }

  /// Function to retrieve the weights for storing purposes.
  Map<String, List<dynamic>> getWeights();

  /// Function to set the weights to the provided values.
  void setWeights(Map<String, List<dynamic>> weights);
}
