/// Holds and exports all the differentLayerTypes that can be used for model building
library tapeLayerTypes;

export 'averagePoolingTapeLayer.dart';
export 'batchNormalizationTapeLayer.dart';
export 'conv2DTapeLayer.dart';
export 'convLSTMTapeLayer.dart';
export 'denseTapeLayer.dart';
export 'dropoutTapeLayer.dart';
export 'dualLSTMTapeLayer.dart';
export 'embeddingTapeLayer.dart';
export 'flattenTapeLayer.dart';
export 'geluTapeLayer.dart';
export 'globalAveragePoolingTapeLayer.dart';
export 'layerNormalizationTapeLayer.dart';
export 'LSTMTapeLayer.dart';
export 'maxPoolingTapeLayer.dart';
export 'multiHeadAttentionTapeLayer.dart';
export 'positionalEncodingTapeLayer.dart';
export 'reluTapeLayer.dart';
export 'rnnTapeLayer.dart';
export 'sigmoidMatrixTapeLayer.dart';
export 'singleHeadAttentionTapeLayer.dart';
export 'transformerEncoderBlockTapeLayer.dart';



import '/tensor/tensor_gpu.dart';
import '../ffi/commandBuffer.dart';

abstract class TapeLayer {
  String get name;

  // Exposes learnable weights and biases to the Optimizer
  List<GPUTensor> get parameters;

  bool built = false;

  // Allocates VRAM for weights/biases based on the incoming input shape
  void build(GPUTensor<dynamic> input) {
    built = true;
  }

  // Appends this layer's forward operations to the shared tape.
  // The intermediates list captures transient tensors to prevent VRAM creep.
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates);

  GPUTensor<dynamic> call(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    if (built == false) {
      build(input);
    }
    return forward(input, tape, intermediates);
  }

  void free() {
    List<GPUTensor> params = parameters;
    for (int i = 0; i < params.length; i = i + 1) {
      params[i].free(); //
    }
  }

  // Used for moving weights back to CPU to save the model to disk
  Map<String, List<dynamic>> getWeights();
  void setWeights(Map<String, List<dynamic>> weights);
}