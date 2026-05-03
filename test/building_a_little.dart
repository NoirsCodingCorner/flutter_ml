

import 'package:flutter_ml/gpu_version/ffi/commandBuffer.dart';
import 'package:flutter_ml/gpu_version/ffi/cudaEngine.dart';
import 'package:flutter_ml/tensor/tensor_gpu.dart';
import 'package:flutter_ml/tensor/tensor_math_gpu.dart';
import 'package:flutter_ml/tensor/type_Aliases.dart';

void main(){
  CudaEngine.initialize(debug: false);
  GPUTensor<Vec> a = GPUTensor<Vec>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
  GPUTensor<Vec> b = GPUTensor<Vec>([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

  CommandBuffer tape= CommandBuffer();
  GPUTensor<Scalar> c = dotProductGPU(a, b, tape);
  GPUTensor<Scalar> d = l2NormGPU(a, tape);
  GPUTensor<Scalar> e = euclideanDistanceGPU(a,b, tape);
  GPUTensor<Scalar> f = cosineSimilarityGPU(a, b, tape);
  GPUTensor<Scalar> g = maeLossGPU(a, b, tape);
  CudaEngine.run(tape.bytes());
  a.toCpu();
  b.toCpu();
  c.toCpu();
  d.toCpu();
  e.toCpu();
  f.toCpu();
  g.toCpu();
  print("Original Tensors: ${a.value}, ${b.value}");
  print("Dot Product: ${c.value}");
  print("L2 Norm: ${d.value}");
  print("Euclidean Distance: ${e.value}");
  print("Cosine Similarity: ${f.value}");
  print("MAE Loss: ${g.value}");
}