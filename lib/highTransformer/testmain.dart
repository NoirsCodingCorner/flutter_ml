

import 'package:flutter_ml/full_library.dart';

void testGemmaOperations(Target target) async {
  await GPUEngine.initialize(debug: false, target: target);

  print("==================================================================");
  print("               TESTING GEMMA LLM GPU OPERATIONS                   ");
  print("==================================================================");

  // ------------------------------------------------------------------
  // 1. RMSNorm
  // ------------------------------------------------------------------
  print("\n--- Testing RMSNorm ---");
  GPUTensor<Matrix> rmsIn = GPUTensor<Matrix>.empty(<int>[2, 4]);
  GPUTensor<Vector> rmsWeight = GPUTensor<Vector>.empty(<int>[4]);

  // Zeile 0: 1, 2, 3, 4 | Zeile 1: -1, -2, -3, -4
  rmsIn.pushData(<double>[
    1.0, 2.0, 3.0, 4.0,
    -1.0, -2.0, -3.0, -4.0
  ]);
  // Gamma Skalierung (Gemma nutzt oft scale_plus_one)
  rmsWeight.pushData(<double>[0.1, 0.2, 0.3, 0.4]);

  CommandBuffer tape1 = CommandBuffer();
  GPUTensor<Matrix> rmsOut = rmsNormMatrixGPU(rmsIn, rmsWeight, 1e-5, true, tape1);
  GPUEngine.run(tape1.bytes());

  rmsOut.toCpu();
  print("RMSNorm Output:");
  print(rmsOut.value);

  rmsIn.free();
  rmsWeight.free();
  rmsOut.free();

  // ------------------------------------------------------------------
  // 2. Causal Mask
  // ------------------------------------------------------------------
  print("\n--- Testing Causal Mask ---");
  GPUTensor<Tensor3D> maskIn = GPUTensor<Tensor3D>.empty(<int>[1, 3, 3]);
  maskIn.pushData(<double>[
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
  ]);

  CommandBuffer tape2 = CommandBuffer();
  GPUTensor<Tensor3D> maskOut = causalMaskGPU(maskIn, 1, 1, 3, tape2);
  GPUEngine.run(tape2.bytes());

  maskOut.toCpu();
  print("Causal Mask Output (Erwartet -10000.0 über der Diagonale):");
  print(maskOut.value);

  maskIn.free();
  maskOut.free();

  // ------------------------------------------------------------------
  // 3. Rotary Position Embeddings (RoPE)
  // ------------------------------------------------------------------
  print("\n--- Testing RoPE ---");
  // batch=1, seq=2, heads=1, headDim=4
  GPUTensor<Tensor3D> ropeIn = GPUTensor<Tensor3D>.empty(<int>[1, 2, 4]);
  ropeIn.pushData(<double>[
    1.0, 2.0, 3.0, 4.0, // Token 0
    1.0, 2.0, 3.0, 4.0  // Token 1
  ]);

  // cos/sin Tabellen: [seqLen, headDim/2] = [2, 2]
  GPUTensor<Matrix> cosTable = GPUTensor<Matrix>.empty(<int>[2, 2]);
  GPUTensor<Matrix> sinTable = GPUTensor<Matrix>.empty(<int>[2, 2]);

  // Token 0: Keine Rotation (cos=1, sin=0)
  // Token 1: 90 Grad Rotation (cos=0, sin=1)
  cosTable.pushData(<double>[1.0, 1.0, 0.0, 0.0]);
  sinTable.pushData(<double>[0.0, 0.0, 1.0, 1.0]);

  CommandBuffer tape3 = CommandBuffer();
  GPUTensor<Tensor3D> ropeOut = applyRopeGPU(ropeIn, cosTable, sinTable, 1, 2, 1, 4, tape3);
  GPUEngine.run(tape3.bytes());

  ropeOut.toCpu();
  print("RoPE Output:");
  print("Token 0 (Sollte gleich bleiben): ${ropeOut.value[0][0]}");
  print("Token 1 (Rotierend zu [-x1, x0, -x3, x2]): ${ropeOut.value[0][1]}");

  ropeIn.free();
  cosTable.free();
  sinTable.free();
  ropeOut.free();

  // ------------------------------------------------------------------
  // 4. Categorical Cross Entropy Loss
  // ------------------------------------------------------------------
  print("\n--- Testing Cross Entropy Loss ---");
  // batch=2, vocab=3
  GPUTensor<Matrix> ceLogits = GPUTensor<Matrix>.empty(<int>[2, 3]);
  ceLogits.pushData(<double>[
    10.0, 1.0, 1.0, // Extrem sicher in Klasse 0
    1.0, 1.0, 10.0  // Extrem sicher in Klasse 2
  ]);
  GPUTensor<Vector> ceTargets = GPUTensor<Vector>.empty(<int>[2]);
  ceTargets.pushData(<double>[0.0, 2.0]);

  CommandBuffer tape4 = CommandBuffer();
  GPUTensor<Vector> ceLoss = crossEntropyLossGPU(ceLogits, ceTargets, tape4);
  GPUEngine.run(tape4.bytes());

  ceLoss.toCpu();
  print("Cross Entropy Loss (Sollte für beide sehr nah an 0.0 sein):");
  print(ceLoss.value);

  ceLogits.free();
  ceTargets.free();
  ceLoss.free();

  // ------------------------------------------------------------------
  // 5. Argmax Sampling
  // ------------------------------------------------------------------
  print("\n--- Testing Argmax ---");
  GPUTensor<Matrix> argmaxLogits = GPUTensor<Matrix>.empty(<int>[2, 3]);
  argmaxLogits.pushData(<double>[
    1.0, 5.0, 2.0, // Maximum bei Index 1
    8.0, 2.0, 9.0  // Maximum bei Index 2
  ]);

  CommandBuffer tape5 = CommandBuffer();
  GPUTensor<Vector> argmaxOut = argmaxGPU(argmaxLogits, tape5);
  GPUEngine.run(tape5.bytes());

  argmaxOut.toCpu();
  print("Argmax Output (Sollte [1.0, 2.0] sein):");
  print(argmaxOut.value);

  argmaxLogits.free();
  argmaxOut.free();

  GPUEngine.dispose();
}


void main(){
  testGemmaOperations(Target.cuda);
}


