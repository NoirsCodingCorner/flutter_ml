import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'tapeLayer.dart';

/// A full Transformer Encoder Block.
/// Combines Multi-Head Self-Attention, Residual Connections, Layer Normalization,
/// and a position-wise Feed-Forward Network (FFN) using GELU activation.
class TransformerEncoderBlockTapeLayer extends TapeLayer<Matrix, Matrix> {
  @override
  String get name => 'TransformerEncoderBlockTapeLayer';

  int dModel;
  int numHeads;
  int dff;

  late MultiHeadAttentionTL mha;
  late LayerNormalizationTL norm1;
  late LayerNormalizationTL norm2;

  /// Expansion weight matrix for the Feed-Forward Network.
  late GPUTensor<Matrix> W1;
  /// Expansion bias vector for the Feed-Forward Network.
  late GPUTensor<Vector> b1;
  /// Projection weight matrix for the Feed-Forward Network.
  late GPUTensor<Matrix> W2;
  /// Projection bias vector for the Feed-Forward Network.
  late GPUTensor<Vector> b2;

  /// Persistent Cache for Static Unrolling
  int cacheSeqLength = -1;
  List<GPUTensor<Matrix>> stepCache = <GPUTensor<Matrix>>[];

  /// Requires the model dimension [dModel], number of attention heads [numHeads],
  /// and the inner hidden size of the feed-forward network [dff] (usually 4x dModel).
  TransformerEncoderBlockTapeLayer(this.dModel, this.numHeads, this.dff) {
    mha = MultiHeadAttentionTL(dModel, numHeads);
    norm1 = LayerNormalizationTL(dModel);
    norm2 = LayerNormalizationTL(dModel);
  }

  /// Returns the trainable parameters for this block, including all parameters from the nested MHA and Norm layers.
  @override
  List<GPUTensor> get parameters {
    List<GPUTensor> params = <GPUTensor>[];
    if (built) {
      params.addAll(mha.parameters);
      params.addAll(norm1.parameters);
      params.addAll(norm2.parameters);
      params.add(W1);
      params.add(b1);
      params.add(W2);
      params.add(b2);
    }
    return params;
  }

  /// Allocates VRAM for child layers and the FFN components.
  @override
  void build(GPUTensor<Matrix> input) {
    mha.build(input);
    norm1.build(input);
    norm2.build(input);

    double scaleW1 = sqrt(2.0 / (dModel + dff));
    W1 = GPUTensor<Matrix>.randomUniform(<int>[dModel, dff], scaleW1);
    b1 = GPUTensor<Vector>.empty(<int>[dff]);

    double scaleW2 = sqrt(2.0 / (dff + dModel));
    W2 = GPUTensor<Matrix>.randomUniform(<int>[dff, dModel], scaleW2);
    b2 = GPUTensor<Vector>.empty(<int>[dModel]);

    built = true;
  }

  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      map.addAll(mha.getNamedParameters('$prefix.attention'));
      map.addAll(norm1.getNamedParameters('$prefix.attention.output.LayerNorm'));
      map.addAll(norm2.getNamedParameters('$prefix.output.LayerNorm'));

      map['$prefix.intermediate.dense.weight'] = W1;
      map['$prefix.intermediate.dense.bias'] = b1;
      map['$prefix.output.dense.weight'] = W2;
      map['$prefix.output.dense.bias'] = b2;
    }
    return map;
  }

  /// Appends the encoder block operations to the [tape].
  /// Relies on the internal persistent caches of its child layers and caches its own FFN math.
  @override
  GPUTensor<Matrix> forward(GPUTensor<Matrix> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    int seqLength = input.shape[0];
    bool useCache = (cacheSeqLength == seqLength);

    if (!useCache) {
      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = seqLength;
    }

    int cIdx = 0;

    GPUTensor<Matrix>? getCached() {
      if (useCache) {
        GPUTensor<Matrix> cached = stepCache[cIdx];
        cIdx = cIdx + 1;
        return cached;
      }
      return null;
    }

    void saveCached(GPUTensor<Matrix> tensor) {
      if (!useCache) {
        stepCache.add(tensor);
      }
    }

    // 1. Multi-Head Attention (MHA manages its own persistent cache)
    GPUTensor<Matrix> attnOut = mha.forward(input, tape, intermediates);

    // 2. Residual Add & Layer Norm 1
    GPUTensor<Matrix> add1 = addMatrixGPU(input, attnOut, tape, outTensor: getCached());
    saveCached(add1);

    GPUTensor<Matrix> norm1Out = norm1.forward(add1, tape, intermediates);

    // 3. FFN Layer 1
    GPUTensor<Matrix> ffn1MatMul = matMulGPU(norm1Out, W1, tape, outTensor: getCached());
    saveCached(ffn1MatMul);

    GPUTensor<Matrix> ffn1Bias = addBiasToMatMulOutGPU(ffn1MatMul, b1, tape, outTensor: getCached());
    saveCached(ffn1Bias);

    GPUTensor<Matrix> ffn1 = geluMatrixGPU(ffn1Bias, tape, outTensor: getCached());
    saveCached(ffn1);

    // 4. FFN Layer 2 (MatMul + Bias)
    GPUTensor<Matrix> ffn2MatMul = matMulGPU(ffn1, W2, tape, outTensor: getCached());
    saveCached(ffn2MatMul);

    GPUTensor<Matrix> ffn2 = addBiasToMatMulOutGPU(ffn2MatMul, b2, tape, outTensor: getCached());
    saveCached(ffn2);

    // 5. Residual Add & Layer Norm 2
    GPUTensor<Matrix> add2 = addMatrixGPU(norm1Out, ffn2, tape, outTensor: getCached());
    saveCached(add2);

    GPUTensor<Matrix> out = norm2.forward(add2, tape, intermediates);

    return out;
  }

  /// Clears the gradients of all nested layers and its own internal FFN caches.
  @override
  void zeroStates(CommandBuffer tape) {
    mha.zeroStates(tape);
    norm1.zeroStates(tape);
    norm2.zeroStates(tape);

    for (int i = 0; i < stepCache.length; i = i + 1) {
      stepCache[i].zeroGrad(tape);
    }
  }

  /// Frees VRAM for child layers, FFN parameters, and its local persistent cache.
  @override
  void free() {
    if (built) {
      mha.free();
      norm1.free();
      norm2.free();
      W1.free();
      b1.free();
      W2.free();
      b2.free();

      for (int i = 0; i < stepCache.length; i = i + 1) {
        stepCache[i].free();
      }
      stepCache.clear();
      cacheSeqLength = -1;
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    Map<String, List<dynamic>> wMap = <String, List<dynamic>>{};
    if (!built) return wMap;

    W1.toCpu();
    b1.toCpu();
    W2.toCpu();
    b2.toCpu();

    wMap['W1'] = W1.value;
    wMap['b1'] = b1.value;
    wMap['W2'] = W2.value;
    wMap['b2'] = b2.value;

    // Flatten child weights to maintain Map<String, List<dynamic>> strict typing
    Map<String, List<dynamic>> mhaWeights = mha.getWeights();
    List<String> mhaKeys = mhaWeights.keys.toList();
    for (int i = 0; i < mhaKeys.length; i = i + 1) {
      wMap['mha.${mhaKeys[i]}'] = mhaWeights[mhaKeys[i]]!;
    }

    Map<String, List<dynamic>> norm1Weights = norm1.getWeights();
    List<String> norm1Keys = norm1Weights.keys.toList();
    for (int i = 0; i < norm1Keys.length; i = i + 1) {
      wMap['norm1.${norm1Keys[i]}'] = norm1Weights[norm1Keys[i]]!;
    }

    Map<String, List<dynamic>> norm2Weights = norm2.getWeights();
    List<String> norm2Keys = norm2Weights.keys.toList();
    for (int i = 0; i < norm2Keys.length; i = i + 1) {
      wMap['norm2.${norm2Keys[i]}'] = norm2Weights[norm2Keys[i]]!;
    }

    return wMap;
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {
    if (built) free();

    List<List<double>> extractMatrix(List<dynamic> raw) {
      List<List<double>> result = <List<double>>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        List<double> row = <double>[];
        List<dynamic> rawRow = raw[i] as List<dynamic>;
        for (int j = 0; j < rawRow.length; j = j + 1) {
          row.add(rawRow[j] as double);
        }
        result.add(row);
      }
      return result;
    }

    List<double> extractVector(List<dynamic> raw) {
      List<double> result = <double>[];
      for (int i = 0; i < raw.length; i = i + 1) {
        result.add(raw[i] as double);
      }
      return result;
    }

    W1 = GPUTensor<Matrix>(extractMatrix(newWeights['W1']!));
    b1 = GPUTensor<Vector>(extractVector(newWeights['b1']!));
    W2 = GPUTensor<Matrix>(extractMatrix(newWeights['W2']!));
    b2 = GPUTensor<Vector>(extractVector(newWeights['b2']!));

    // Extract child weights based on the flattened prefixes
    Map<String, List<dynamic>> mhaWeights = <String, List<dynamic>>{};
    Map<String, List<dynamic>> norm1Weights = <String, List<dynamic>>{};
    Map<String, List<dynamic>> norm2Weights = <String, List<dynamic>>{};

    List<String> allKeys = newWeights.keys.toList();
    for (int i = 0; i < allKeys.length; i = i + 1) {
      String key = allKeys[i];
      if (key.startsWith('mha.')) {
        mhaWeights[key.substring(4)] = newWeights[key]!;
      } else if (key.startsWith('norm1.')) {
        norm1Weights[key.substring(6)] = newWeights[key]!;
      } else if (key.startsWith('norm2.')) {
        norm2Weights[key.substring(6)] = newWeights[key]!;
      }
    }

    mha.setWeights(mhaWeights);
    norm1.setWeights(norm1Weights);
    norm2.setWeights(norm2Weights);

    built = true;
  }
}