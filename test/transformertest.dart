import 'package:flutter_ml_web_gpu/gpu_version/SeqModel.dart';
import 'package:flutter_ml_web_gpu/gpu_version/optimizer/adam.dart';
import 'package:flutter_ml_web_gpu/logger.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_ml_web_gpu/full_library.dart';

void main() {
  GPUEngine.initialize(target: Target.cuda);
  test('Minimalist Transformer Sequence-to-Sequence', () {
    // 1. Setup Architecture Dimensions
    int vocabSize = 10;
    int dModel = 16;
    int numHeads = 4;
    int dff = 32;
    int maxSeqLength = 8;

    // 2. Training Data: A sequence of 4 token IDs
    GPUTensor<Vector> trainInput = GPUTensor<Vector>(<double>[1.0, 3.0, 5.0, 2.0]);

    // Training Target: One-hot encoded matrix for the target outputs (Shape: 4 x 10)
    GPUTensor<Matrix> trainTarget = GPUTensor<Matrix>(<List<double>>[
      <double>[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 1.0 -> 2.0
      <double>[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 3.0 -> 4.0
      <double>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // Target for input 5.0 -> 6.0
      <double>[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Target for input 2.0 -> 3.0
    ]);

    // 3. Define the Transformer Pipeline
    List<TapeLayer> layers = <TapeLayer>[
      // Embeds the 1D token IDs into a 2D Matrix of shape [SeqLength, dModel]
      EmbeddingTL(vocabSize, dModel),

      // Injects sine/cosine spatial frequencies
      PositionalEncodingTL(maxSeqLength, dModel),

      // The heavy-lifter: Multi-Head Self Attention + Residuals + LayerNorm + FFN
      TransformerEncoderBlockTapeLayer(dModel, numHeads, dff),

      // Projects the 16-dimensional features back into the 10-dimensional vocabulary space
      DenseTL(vocabSize*16),
    ];

    // 4. Initialize the Sequential Wrapper
    SeqModel<Vector, Matrix> model = SeqModel<Vector, Matrix>(
      layers,
      trainInput,
      target: trainTarget,
      lossFunction: mseMatrixGPU,
      optimizerBuilder: (List<GPUTensor> params) {
        return AdamGPU(params, 0.01);
      },
    );

    Logger.blue('Compiling Transformer Tapes...');
    model.compile();
    Logger.green('Compilation Successful.');

    // 5. Run the Training Loop
    int epochs = 250;
    Logger.blue('Starting Training for $epochs epochs...');

    for (int i = 0; i < epochs; i = i + 1) {
      model.runTraining();
    }

    model.loss?.toCpu();
    Logger.green('Final Training Loss: ${model.loss?.value}');

    // 6. Test Static Inference with dynamic sequence lengths!
    // Notice how the inference sequence length (2) is different from training (4).
    // The model.predict() gracefully handles reshaping the internal caches.
    GPUTensor<Vector> inferInput = GPUTensor<Vector>(<double>[1.0, 3.0]);
    GPUTensor<Matrix> inferResult = model.predict(inferInput);

    model.runForward();
    inferResult.toCpu();

    Logger.blue('--- Inference Results ---');
    Logger.blue('Input Tokens: [1.0, 3.0]');

    // Prints the output probabilities/logits for the 2 tokens across the 10 vocab classes
    List<List<double>> outputMatrix = inferResult.value;
    for (int i = 0; i < outputMatrix.length; i = i + 1) {
      Logger.blue('Token $i Output Logits: ${outputMatrix[i]}');
    }

    model.free();
  });
}