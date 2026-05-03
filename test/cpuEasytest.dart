
import 'package:flutter_ml/full_library.dart';

void main(){
  Tensor<Vector>VecA=Tensor([1.1, 2.2, 3.3]);
  Tensor<Vector>VecB=Tensor([4.4, 5.5, 6.6]);
  Tensor<Vector>VecC=Tensor([1.0,0.1,-1.0]);

  Tensor<Vector>VecD=addVector(VecA, VecB);
  Tensor<Vector>VecE=elementWiseMultiply(VecC, VecD);
  print(VecE.value);
  VecE.printGraph();
}