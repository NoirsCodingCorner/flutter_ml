import 'dart:math';

import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../layertypes/layer.dart';
import 'embeddingLayer.dart';

/// Injects information about the relative or absolute position of tokens in a sequence.
///
/// Since the Transformer architecture contains no recurrence, it has no inherent
/// sense of word order. This layer adds a unique, non-trainable vector to each
/// input embedding, allowing the model to learn from the sequence order.
///
/// It uses the standard sinusoidal formula from the "Attention Is All You Need" paper.
class PositionalEncoding extends Layer {
  @override
  String name = 'positional_encoding';
  int maxLength;
  int dModel;

  late Tensor<Matrix> encodingMatrix;

  PositionalEncoding(this.maxLength, this.dModel);

  @override
  List<Tensor> get parameters => [];

  @override
  void build(Tensor<dynamic> input) {
    Matrix pe = [];
    for (int i = 0; i < maxLength; i++) {
      Vector row = [];
      for (int j = 0; j < dModel; j++) {
        row.add(0.0);
      }
      pe.add(row);
    }

    for (int pos = 0; pos < maxLength; pos++) {
      for (int i = 0; i < dModel; i++) {
        double angle = pos / pow(10000, (2 * i) / dModel);
        if (i % 2 == 0) {
          pe[pos][i] = sin(angle);
        } else {
          pe[pos][i] = cos(angle);
        }
      }
    }
    encodingMatrix = Tensor<Matrix>(pe);
    super.build(input);
  }

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;
    int sequenceLength = inputMatrix.value.length;

    Matrix applicableEncodings = [];
    for(int i=0; i < sequenceLength; i++){
      applicableEncodings.add(encodingMatrix.value[i]);
    }
    Tensor<Matrix> positionalTensor = Tensor<Matrix>(applicableEncodings);

    return addMatrix(inputMatrix, positionalTensor);
  }
}

/*void main() {
  int vocabSize = 1000;
  int dModel = 128;
  int maxSequenceLength = 50;

  // 1. Define the model
  SNetwork model = SNetwork([
    EmbeddingLayer(vocabSize, dModel),
    PositionalEncoding(maxSequenceLength, dModel),
    // ... other layers like Transformer Blocks would follow ...
  ]);

  // 2. Build the model with a dummy input
  // The input must match what the FIRST layer expects (a Vector of indices).
  Tensor<Vector> dummySentence = Tensor<Vector>([0]);
  model.predict(dummySentence);
  print('Model built successfully.');

  // 3. Run a real sample through the model
  Tensor<Vector> sentence = Tensor<Vector>([10, 42, 5, 99]); // A 4-word sentence
  Tensor<Matrix> finalEmbeddings = model.predict(sentence) as Tensor<Matrix>;

  print('\nInput sentence has ${sentence.value.length} words.');
  print('Final output shape is [${finalEmbeddings.value.length}, ${finalEmbeddings.value[0].length}]');
  print('This confirms the positional information was added correctly.');
}*/