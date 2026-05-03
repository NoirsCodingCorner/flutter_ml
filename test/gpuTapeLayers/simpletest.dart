import 'package:flutter_ml/full_library.dart';

void main() async{
  await CudaEngine.initialize(debug: false);
  CommandBuffer tape=CommandBuffer();

  GPUTensor<Vector>VecA=GPUTensor([1.1, 2.2, 3.3]);
  GPUTensor<Vector>VecB=GPUTensor([4.4, 5.5, 6.6]);
  GPUTensor<Vector>VecC=GPUTensor([1.0,0.1,-1.0]);

  GPUTensor<Vector>VecD=addVectorGPU(VecA, VecB, tape);
  GPUTensor<Vector>VecE=elementWiseMultiplyGPU(VecC, VecD, tape);

  CudaEngine.run(tape.bytes());


  VecE.toCpu();
  print("Result: ${VecE.value}");
  VecE.printGraph();
  TapeDecoder(tape.bytes()).decode();
}
