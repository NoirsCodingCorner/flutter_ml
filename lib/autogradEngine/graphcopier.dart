import 'package:flutter_ml/autogradEngine/tensor.dart';

class GraphCopier {
  final Map<Tensor, Tensor> _oldToNewTensorMap = {};

  Tensor copy(Tensor finalTensor) {
    _oldToNewTensorMap.clear();
    return _recursiveCopy(finalTensor);
  }

  Tensor _recursiveCopy(Tensor originalTensor) {
    if (_oldToNewTensorMap.containsKey(originalTensor)) {
      return _oldToNewTensorMap[originalTensor]!;
    }

    if (originalTensor.creator == null) {
      final newLeaf = Tensor(originalTensor.value);
      _oldToNewTensorMap[originalTensor] = newLeaf;
      return newLeaf;
    }

    Node originalNode = originalTensor.creator!;
    List<Tensor> newInputs = [];
    for (Tensor originalInput in originalNode.inputs) {
      newInputs.add(_recursiveCopy(originalInput));
    }

    Tensor newTensor;
    // This switch statement now uses the unique opName for each function.
    switch (originalNode.opName) {
    // SCALAR OPS
      case 'multiply_scalar':
        newTensor = multiply(newInputs[0] as Tensor<Scalar>, newInputs[1] as Tensor<Scalar>);
        break;
      case 'sigmoidScalar':
        newTensor = sigmoidScalar(newInputs[0] as Tensor<Scalar>);
        break;
      case 'binaryCrossEntropy':
        newTensor = binaryCrossEntropy(newInputs[0] as Tensor<Scalar>, newInputs[1] as Tensor<Scalar>);
        break;

    // VECTOR OPS
      case 'add_vector':
        newTensor = add(newInputs[0] as Tensor<Vector>, newInputs[1] as Tensor<Vector>);
        break;
      case 'addScalar_vector':
        double s = originalNode.extraParams['s'];
        newTensor = addScalar(newInputs[0] as Tensor<Vector>, s);
        break;
      case 'concat_vector':
        newTensor = concatenate(newInputs[0] as Tensor<Vector>, newInputs[1] as Tensor<Vector>);
        break;
      case 'dot':
        newTensor = dot(newInputs[0] as Tensor<Vector>, newInputs[1] as Tensor<Vector>);
        break;
      case 'multiply_vector':
        newTensor = elementWiseMultiply(newInputs[0] as Tensor<Vector>, newInputs[1] as Tensor<Vector>);
        break;
      case 'mse_vector':
        newTensor = mse(newInputs[0] as Tensor<Vector>, newInputs[1] as Tensor<Vector>);
        break;
      case 'relu_vector':
        newTensor = relu(newInputs[0] as Tensor<Vector>);
        break;
      case 'sigmoid_vector':
        newTensor = sigmoid(newInputs[0] as Tensor<Vector>);
        break;
      case 'sum_vector':
        newTensor = sum(newInputs[0] as Tensor<Vector>);
        break;
      case 'tanh_vector':
        newTensor = vectorTanh(newInputs[0] as Tensor<Vector>);
        break;
      case 'exp_vector':
        newTensor = vectorExp(newInputs[0] as Tensor<Vector>);
        break;
      case 'log_vector':
        newTensor = vectorLog(newInputs[0] as Tensor<Vector>);
        break;

    // MATRIX OPS
      case 'add_matrix':
        newTensor = addMatrix(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Matrix>);
        break;
      case 'addMatrixAndVector':
        newTensor = addMatrixAndVector(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Vector>);
        break;
      case 'addScalarToMatrix':
        newTensor = addScalarToMatrix(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Scalar>);
        break;
      case 'concat_matrix_col':
        newTensor = concatenateMatricesByColumn(newInputs.cast<Tensor<Matrix>>());
        break;
      case 'conv2d':
        String padding = originalNode.extraParams['padding'];
        newTensor = conv2d(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Matrix>, padding: padding);
        break;
      case 'multiply_matrix':
        newTensor = elementWiseMultiplyMatrix(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Matrix>);
        break;
      case 'matMul':
        newTensor = matMul(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Matrix>);
        break;
      case 'matVecMul':
        newTensor = matVecMul(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Vector>);
        break;
      case 'mse_matrix':
        newTensor = mseMatrix(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Matrix>);
        break;
      case 'relu_matrix':
        newTensor = reluMatrix(newInputs[0] as Tensor<Matrix>);
        break;
      case 'reshape':
        int numRows = originalNode.extraParams['numRows'];
        int numCols = originalNode.extraParams['numCols'];
        newTensor = reshapeVectorToMatrix(newInputs[0] as Tensor<Vector>, numRows, numCols);
        break;
      case 'scale_matrix':
        double s = originalNode.extraParams['s'];
        newTensor = scaleMatrix(newInputs[0] as Tensor<Matrix>, s);
        break;
      case 'selectRow':
        int rowIndex = originalNode.extraParams['rowIndex'];
        newTensor = selectRow(newInputs[0] as Tensor<Matrix>, rowIndex);
        break;
      case 'sigmoid_matrix':
        newTensor = sigmoidMatrix(newInputs[0] as Tensor<Matrix>);
        break;
      case 'softmax_matrix':
        newTensor = softmaxMatrix(newInputs[0] as Tensor<Matrix>);
        break;
      case 'sum_matrix':
        newTensor = sumMatrix(newInputs[0] as Tensor<Matrix>);
        break;
      case 'tanh_matrix':
        newTensor = tanhMatrix(newInputs[0] as Tensor<Matrix>);
        break;
      case 'transpose':
        newTensor = transpose(newInputs[0] as Tensor<Matrix>);
        break;

    // 3D OPS
      case 'add_3d':
        newTensor = add3D(newInputs[0] as Tensor<Tensor3D>, newInputs[1] as Tensor<Tensor3D>);
        break;
      case 'multiply_3d':
        newTensor = elementWiseMultiply3D(newInputs[0] as Tensor<Tensor3D>, newInputs[1] as Tensor<Tensor3D>);
        break;
      case 'concat_3d':
        newTensor = concatenate3D(newInputs[0] as Tensor<Tensor3D>, newInputs[1] as Tensor<Tensor3D>);
        break;

    // COMPOSITE OPS
      case 'layer_norm':
        double epsilon = originalNode.extraParams['epsilon'];
        newTensor = layerNorm(newInputs[0] as Tensor<Matrix>, newInputs[1] as Tensor<Vector>, newInputs[2] as Tensor<Vector>, epsilon: epsilon);
        break;

      default:
        throw UnimplementedError("Copy operation for '${originalNode.opName}' is not defined.");
    }

    _oldToNewTensorMap[originalTensor] = newTensor;
    return newTensor;
  }
}

// Example usage remains the same and now works correctly.
void main() {
  // 1. Create an original computational graph
  Tensor<Scalar> a = Tensor<Scalar>(2.0);
  Tensor<Scalar> b = Tensor<Scalar>(3.0);
  Tensor<Scalar> c = multiply(a, b); // This now correctly uses opName: 'multiply_scalar'
  Tensor<Scalar> d = multiply(a, c);
  print('--- Original Graph ---');
  d.printGraph();

  // 2. Create a GraphCopier instance
  GraphCopier copier = GraphCopier();

  // 3. Copy the graph starting from the final tensor 'd'
  Tensor d_copy = copier.copy(d);
  print('\n--- Copied Graph ---');
  d_copy.printGraph();

  // 4. Verify they are independent
  print('\nOriginal final tensor hash: ${d.hashCode}');
  print('Copied final tensor hash:   ${d_copy.hashCode}');

  // Performing backward on one does not affect the other.
  d.backward();
  print('\nOriginal "a" gradient after backward: ${a.grad}');
}