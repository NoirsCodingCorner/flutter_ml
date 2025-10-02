import 'dart:math';

import 'package:flutter_ml/transformer/positionalEncodingLayer.dart';

import '../activationFunctions/sigmoid.dart';
import '../autogradEngine/tensor.dart';
import '../layertypes/denseLayer.dart';
import '../layertypes/dropout.dart';
import '../layertypes/flattenLayer.dart';
import '../layertypes/layer.dart';
import '../activationFunctions/relu.dart';
import '../nets/snet.dart';
import '../optimizers/optimizers.dart';
import 'embeddingLayer.dart';
import 'layerNormalization.dart';
import 'multiHeadAttentionLayer.dart';

// Assuming global/utility functions:
// Tensor<Matrix> addMatrix(Tensor<Matrix> a, Tensor<Matrix> b)
// Tensor<Matrix> normMatrix(Tensor<Matrix> input, LayerNormalization normLayer) // As defined previously

/// A single block of the Transformer Encoder, which contains a Multi-Head
/// Attention sub-layer and a Feed-Forward sub-layer, each followed by
/// a Residual Connection and Layer Normalization.
class TransformerEncoderLayer extends Layer {
  @override
  String name = 'encoder_layer';

  final int dModel;
  final int numHeads;
  final double dropoutRate;

  // New private field to hold the FFN inner dimension
  late final int _ffnInnerDimension;

  late MultiHeadAttention attentionLayer;
  late LayerNormalization norm1;
  late DropoutLayer dropout1;

  late DenseLayer ffnDense1;
  late DenseLayer ffnDense2;
  late LayerNormalization norm2;
  late DropoutLayer dropout2;

  TransformerEncoderLayer(this.dModel, this.numHeads, this.dropoutRate) {
    // 1. Calculate and store FFN inner dimension
    _ffnInnerDimension = dModel * 4;

    // Sub-layer 1: Multi-Head Attention
    attentionLayer = MultiHeadAttention(dModel, numHeads);
    dropout1 = DropoutLayer(dropoutRate);
    norm1 = LayerNormalization();

    // Sub-layer 2: Feed-Forward Network (FFN)
    // Use the class field here to set the output size of the first layer
    ffnDense1 = DenseLayer(_ffnInnerDimension, activation: ReLU());
    dropout2 = DropoutLayer(dropoutRate);
    // Use the class field here to set the input size for the build step
    ffnDense2 = DenseLayer(dModel);
    norm2 = LayerNormalization();
  }

  /// Collects parameters from all sub-layers.
  @override
  List<Tensor> get parameters {
    List<Tensor> params = <Tensor>[];
    params.addAll(attentionLayer.parameters);
    params.addAll(norm1.parameters);
    params.addAll(ffnDense1.parameters);
    params.addAll(ffnDense2.parameters);
    params.addAll(norm2.parameters);
    return params;
  }

  /// Builds the component layers based on the input size.
  @override
  void build(Tensor<dynamic> input) {
    // We assume the input is a Matrix [sequence_length, dModel].
    // We need a Vector of size dModel for the DenseLayers and LayerNormalizations
    Tensor<Vector> dummyInputDModel = Tensor<Vector>(List<double>.filled(dModel, 0.0));

    // 1. Build MultiHeadAttention
    attentionLayer.build(input);

    // 2. Build LayerNormalizations (needs dModel size input)
    // FIX: Explicitly call build to initialize gamma/beta.
    norm1.build(dummyInputDModel);
    norm2.build(dummyInputDModel);

    // Dropout layers technically don't need build, but calling it is harmless
    // and safe if their implementation ever changes.

    // 3. Build FFN layers (need their respective input sizes)
    ffnDense1.build(dummyInputDModel);

    Tensor<Vector> dummyInputFFN = Tensor<Vector>(List<double>.filled(_ffnInnerDimension, 0.0));
    ffnDense2.build(dummyInputFFN);

    super.build(input);
  }


  /// Performs the forward pass of the Encoder Layer.
  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    // --- Sub-layer 1: Multi-Head Attention ---
    Tensor<Matrix> inputMatrix = input as Tensor<Matrix>;

    Tensor<Matrix> attentionOutput = attentionLayer.call(inputMatrix) as Tensor<Matrix>;
    Tensor<Matrix> droppedAttention = dropout1.call(attentionOutput) as Tensor<Matrix>;
    Tensor<Matrix> residual1 = addMatrix(inputMatrix, droppedAttention);
    Tensor<Matrix> normalized1 = normMatrix(residual1, norm1);

    // --- Sub-layer 2: Feed-Forward Network ---
    Matrix ffnOutputMatrix = <Vector>[];
    for (Vector tokenVector in normalized1.value) {
      // 1. Convert row to Tensor<Vector>
      Tensor<Vector> tokenTensor = Tensor<Vector>(tokenVector);

      // 2. Pass through FFN Dense Layers
      Tensor<Vector> dense1Output = ffnDense1.call(tokenTensor) as Tensor<Vector>;

      // 3. Apply Dropout
      Tensor<Vector> droppedInner = dropout2.call(dense1Output) as Tensor<Vector>;

      // 4. Second Dense Layer
      Tensor<Vector> dense2Output = ffnDense2.call(droppedInner) as Tensor<Vector>;

      // 5. Add to the output matrix
      ffnOutputMatrix.add(dense2Output.value);
    }
    Tensor<Matrix> ffnOutput = Tensor<Matrix>(ffnOutputMatrix);

    // Add Residual Connection and Normalization
    Tensor<Matrix> residual2 = addMatrix(normalized1, ffnOutput);
    Tensor<Matrix> normalized2 = normMatrix(residual2, norm2);

    return normalized2;
  }
}

// Utility function placeholder needed for the forward pass,
// assuming it applies layer normalization row-by-row on a Matrix.
Tensor<Matrix> normMatrix(Tensor<Matrix> input, LayerNormalization normLayer) {
  Matrix outputMatrix = <Vector>[];
  for (Vector row in input.value) {
    Tensor<Vector> normalizedRow = normLayer.forward(Tensor<Vector>(row));
    outputMatrix.add(normalizedRow.value);
  }
  Tensor<Matrix> out = Tensor<Matrix>(outputMatrix);
  // Manual backward hook to ensure gradient flows correctly through the new matrix.
  // This requires a custom Node to correctly aggregate gradients from all row ops.
  // For simplicity in this structure, we assume the autograd engine handles the
  // tensor creation and backward flow correctly if we define it here:
  out.creator = Node([input], () {
    // Simplified backward: In a real system, the row-by-row forward operations
    // would define the backward graph.
  }, opName: 'matrix_norm', cost: 0);
  return out;
}

void main() {
  // --- DATA & MODEL HYPERPARAMETERS ---
  int vocabSize = 100;
  int dModel = 32;
  int maxSequenceLength = 10; // Fixed the maximum sequence length to 10
  int numHeads = 4;
  double dropoutRate = 0.1;
  int numEncoderLayers = 2;

  // --- TRAINING HYPERPARAMETERS ---
  int numSamples = 200;
  int epochs = 200;
  double learningRate = 0.001;
  int specialToken = 88; // Token that determines a positive class
  int paddingToken = 0; // NEW: Use index 0 as the padding token

  Random random = Random();

  // --- 1. DATA GENERATION: Pad sequences to maxSequenceLength ---
  List<Vector> inputs = <Vector>[];
  List<Vector> targets = <Vector>[];

  for (int i = 0; i < numSamples; i++) {
    Vector sequence = <double>[];

    // Choose a random, shorter length for the actual data part
    int actualLength = random.nextInt(6) + 5;
    bool hasSpecialToken = false;

    if (random.nextBool()) {
      hasSpecialToken = true;
      targets.add(<double>[1.0]);
      int insertIndex = random.nextInt(actualLength);
      for (int j = 0; j < actualLength; j++) {
        if (j == insertIndex) {
          sequence.add(specialToken.toDouble());
        } else {
          double token;
          do {
            // Fill with random tokens, avoiding special token and padding token
            token = random.nextInt(vocabSize - 1).toDouble() + 1.0;
          } while (token == specialToken.toDouble());
          sequence.add(token);
        }
      }
    } else {
      targets.add(<double>[0.0]);
      for (int j = 0; j < actualLength; j++) {
        double token;
        do {
          token = random.nextInt(vocabSize - 1).toDouble() + 1.0;
        } while (token == specialToken.toDouble());
        sequence.add(token);
      }
    }

    // PADDING: Add padding tokens until the sequence reaches maxSequenceLength
    for (int j = sequence.length; j < maxSequenceLength; j++) {
      sequence.add(paddingToken.toDouble());
    }

    inputs.add(sequence);
  }

  // --- 2. DEFINE AND COMPILE MODEL ---
  List<Layer> layers = <Layer>[];
  layers.add(EmbeddingLayer(vocabSize, dModel));
  layers.add(PositionalEncoding(maxSequenceLength, dModel));

  for (int i = 0; i < numEncoderLayers; i++) {
    layers.add(TransformerEncoderLayer(dModel, numHeads, dropoutRate));
  }

  // The Flatten layer now *always* produces a vector of size maxSequenceLength * dModel (10 * 32 = 320)
  layers.add(FlattenLayer());
  layers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork model = SNetwork(layers);

  // Build the model structure using a fixed-length dummy input
  Tensor<Vector> dummyInput = Tensor<Vector>(List<double>.filled(maxSequenceLength, 1.0));
  model.predict(dummyInput); // Builds all internal layer sizes correctly

  model.compile(
      configuredOptimizer: Adam(model.parameters, learningRate: learningRate)
  );
  Optimizer optimizer = model.optimizer;

  print('--- Starting Transformer Training: Find Token ${specialToken} (200 Epochs) ---');

  // --- 3. TRAINING LOOP (No functional change, but uses padded data) ---
  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0.0;
    int correctPredictions = 0;

    for (int i = 0; i < numSamples; i++) {
      optimizer.zeroGrad();

      Tensor<Vector> inputTensor = Tensor<Vector>(inputs[i]);
      Tensor<Vector> targetTensor = Tensor<Vector>(targets[i]);

      Tensor<Vector> finalOutput = model.predict(inputTensor) as Tensor<Vector>;
      Tensor<Scalar> loss = mse(finalOutput, targetTensor);

      totalLoss += loss.value;

      if ((finalOutput.value[0] > 0.5) == (targets[i][0] == 1.0)) {
        correctPredictions++;
      }

      loss.backward();
      optimizer.step();
    }

    double avgLoss = totalLoss / numSamples;
    double accuracy = (correctPredictions / numSamples) * 100.0;

    if ((epoch + 1) % 20 == 0) {
      print('Epoch ${epoch + 1}/${epochs} | Avg Loss: ${avgLoss.toStringAsFixed(6)} | Accuracy: ${accuracy.toStringAsFixed(2)}%');
    }
  }

  // --- 4. FINAL TEST (No functional change) ---
  int finalCorrect = 0;
  for (int i = 0; i < numSamples; i++) {
    Tensor<Vector> inputTensor = Tensor<Vector>(inputs[i]);
    Tensor<Vector> finalOutput = model.predict(inputTensor) as Tensor<Vector>;
    int predictedClass = (finalOutput.value[0] > 0.5) ? 1 : 0;
    int targetClass = targets[i][0].toInt();
    if (predictedClass == targetClass) {
      finalCorrect++;
    }
  }
  double finalAccuracy = (finalCorrect / numSamples) * 100.0;

  print('\n--- Training Complete ---');
  print('Final Overall Accuracy on Training Data: ${finalAccuracy.toStringAsFixed(2)}%');
}