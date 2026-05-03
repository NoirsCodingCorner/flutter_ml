import 'dart:math';

import '/tensor/tensor_gpu.dart';
import '/tensor/tensor_math_gpu.dart';
import '/tensor/type_Aliases.dart';

import '../ffi/commandBuffer.dart';
import 'multiHeadAttentionTapeLayer.dart';
import 'tapeLayer.dart';

class TransformerEncoderBlockTapeLayer extends TapeLayer {
  int dModel;
  int numHeads;
  int dff;

  late MultiHeadAttentionTL mha;
  late LayerNormalizationTL norm1;
  late LayerNormalizationTL norm2;

  late GPUTensor<Matrix> W1;
  late GPUTensor<Vector> b1;
  late GPUTensor<Matrix> W2;
  late GPUTensor<Vector> b2;

  TransformerEncoderBlockTapeLayer(this.dModel, this.numHeads, this.dff) {
    mha = MultiHeadAttentionTL(dModel, numHeads);
    norm1 = LayerNormalizationTL(dModel);
    norm2 = LayerNormalizationTL(dModel);
  }

  @override
  String get name {
    return 'TransformerEncoderBlockTapeLayer';
  }

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

  @override
  void build(GPUTensor<dynamic> input) {
    mha.build(input);
    norm1.build(input);
    norm2.build(input);

    // ⚡ Instant GPU-side VRAM allocation & Randomization!
    double scaleW1 = sqrt(2.0 / (dModel + dff));
    W1 = GPUTensor<Matrix>.randomUniform(<int>[dModel, dff], scaleW1);
    b1 = GPUTensor<Vector>.empty(<int>[dff]);

    double scaleW2 = sqrt(2.0 / (dff + dModel));
    W2 = GPUTensor<Matrix>.randomUniform(<int>[dff, dModel], scaleW2);
    b2 = GPUTensor<Vector>.empty(<int>[dModel]);

    built = true;
  }

  // Magic Mapping Function
  Map<String, GPUTensor> getNamedParameters(String prefix) {
    Map<String, GPUTensor> map = <String, GPUTensor>{};
    if (built) {
      // Prefix is usually 'bert.encoder.layer.0'
      map.addAll(mha.getNamedParameters(prefix + '.attention'));
      map.addAll(norm1.getNamedParameters(prefix + '.attention.output.LayerNorm'));
      map.addAll(norm2.getNamedParameters(prefix + '.output.LayerNorm'));

      map[prefix + '.intermediate.dense.weight'] = W1;
      map[prefix + '.intermediate.dense.bias'] = b1;
      map[prefix + '.output.dense.weight'] = W2;
      map[prefix + '.output.dense.bias'] = b2;
    }
    return map;
  }

  @override
  GPUTensor<dynamic> forward(GPUTensor<dynamic> input, CommandBuffer tape, List<GPUTensor> intermediates) {
    GPUTensor<Matrix> typedInput = input as GPUTensor<Matrix>;

    // 1. Multi-Head Attention
    GPUTensor<Matrix> attnOut = mha.forward(typedInput, tape, intermediates) as GPUTensor<Matrix>;

    // 2. Residual Add & Layer Norm 1
    GPUTensor<Matrix> add1 = addMatrixGPU(typedInput, attnOut, tape);
    intermediates.add(add1);
    GPUTensor<Matrix> norm1Out = norm1.forward(add1, tape, intermediates) as GPUTensor<Matrix>;

    // 3. FFN Layer 1 (Un-fused to use GELU!)
    GPUTensor<Matrix> ffn1MatMul = matMulGPU(norm1Out, W1, tape);
    intermediates.add(ffn1MatMul);
    GPUTensor<Matrix> ffn1Bias = addBiasToMatMulOutGPU(ffn1MatMul, b1, tape);
    intermediates.add(ffn1Bias);

    // Applying the GELU kernel you just compiled
    GPUTensor<Matrix> ffn1 = geluMatrixGPU(ffn1Bias, tape);
    intermediates.add(ffn1);

    // 4. FFN Layer 2 (MatMul + Bias)
    GPUTensor<Matrix> ffn2MatMul = matMulGPU(ffn1, W2, tape);
    intermediates.add(ffn2MatMul);
    GPUTensor<Matrix> ffn2 = addBiasToMatMulOutGPU(ffn2MatMul, b2, tape);
    intermediates.add(ffn2);

    // 5. Residual Add & Layer Norm 2
    GPUTensor<Matrix> add2 = addMatrixGPU(norm1Out, ffn2, tape);
    intermediates.add(add2);
    GPUTensor<Matrix> out = norm2.forward(add2, tape, intermediates) as GPUTensor<Matrix>;

    return out;
  }

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
    }
  }

  @override
  Map<String, List<dynamic>> getWeights() {
    return <String, List<dynamic>>{};
  }

  @override
  void setWeights(Map<String, List<dynamic>> newWeights) {}
}