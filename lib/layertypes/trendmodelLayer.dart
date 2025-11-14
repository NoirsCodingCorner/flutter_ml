import 'dart:io';
import 'dart:math';
import 'package:flutter_ml/layertypes/reluLayer.dart';
import 'package:flutter_ml/optimizers/optimizers.dart';
import 'package:flutter_ml/optimizers/sgd.dart';

import '../activationFunctions/relu.dart';
import '../autogradEngine/tensor.dart';
import '../nets/snet.dart';
import '../optimizers/adam.dart';
import 'denseLayer.dart';
import 'dropout.dart';
import 'layer.dart';
import 'lstmLayer.dart';

class GeneralizedChainedScaleLayer extends Layer {
  @override
  String name = 'generalized_chained_scale';

  final int hiddenSize;
  final List<int> grainingSizes;
  final int finalSequenceLength;
  final int numTiers;

  late List<Tensor<Matrix>> lstmWf, lstmWi, lstmWc, lstmWo;
  late List<Tensor<Vector>> lstmBf, lstmBi, lstmBc, lstmBo;
  late List<Tensor<Matrix>> aggW;
  late List<Tensor<Vector>> aggB;

  GeneralizedChainedScaleLayer({
    required this.hiddenSize,
    required this.grainingSizes,
    this.finalSequenceLength = 7,
  }) : numTiers = grainingSizes.length + 1;

  @override
  List<Tensor> get parameters {
    List<Tensor> allParams = [];
    for (int i = 0; i < numTiers; i++) {
      allParams.addAll([
        lstmWf[i], lstmBf[i], lstmWi[i], lstmBi[i],
        lstmWc[i], lstmBc[i], lstmWo[i], lstmBo[i],
      ]);
    }
    for (int i = 0; i < grainingSizes.length; i++) {
      allParams.addAll([aggW[i], aggB[i]]);
    }
    return allParams;
  }

  @override
  void build(Tensor<dynamic> input) {
    Matrix inputMatrix = input.value as Matrix;
    int inputSize = inputMatrix.isNotEmpty ? inputMatrix[0].length : 0;
    Random random = Random();

    lstmWf = []; lstmWi = []; lstmWc = []; lstmWo = [];
    lstmBf = []; lstmBi = []; lstmBc = []; lstmBo = [];
    aggW = []; aggB = [];

    Tensor<Matrix> initWeights(int fanIn, int fanOut) {
      double stddev = sqrt(1.0 / fanIn);
      Matrix values = [];
      for (int i = 0; i < fanOut; i++) {
        Vector row = [];
        for (int j = 0; j < fanIn; j++) {
          row.add((random.nextDouble() * 2 - 1) * stddev);
        }
        values.add(row);
      }
      return Tensor<Matrix>(values);
    }

    int currentAggInputSize = inputSize;
    for (int grainSize in grainingSizes) {
      int fanIn = grainSize * currentAggInputSize;
      aggW.add(initWeights(fanIn, currentAggInputSize));
      aggB.add(Tensor<Vector>(List<double>.filled(currentAggInputSize, 0.0)));
    }

    for (int i = 0; i < numTiers; i++) {
      int lstmInputFeatureSize = inputSize;
      int lstmCombinedSize = hiddenSize + lstmInputFeatureSize;
      lstmWf.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWi.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWc.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmWo.add(initWeights(lstmCombinedSize, hiddenSize));
      lstmBf.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBi.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBc.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
      lstmBo.add(Tensor<Vector>(List<double>.filled(hiddenSize, 0.0)));
    }

    super.build(input);
  }

  @override
  Tensor<Vector> forward(Tensor<dynamic> input) {
    Matrix fullSequence = (input as Tensor<Matrix>).value;

    List<List<Tensor<Vector>>> tierInputs = List.generate(numTiers, (_) => []);

    int lastTierStart = fullSequence.length - finalSequenceLength;
    Matrix finalDataRaw = fullSequence.sublist(lastTierStart);
    tierInputs[0] = finalDataRaw.map((v) => Tensor<Vector>(v)).toList();

    List<Tensor<Vector>> currentLevelData = fullSequence.map((v) => Tensor<Vector>(v)).toList();
    for (int i = 0; i < grainingSizes.length; i++) {
      int grainSize = grainingSizes[i];
      List<Tensor<Vector>> nextLevelSummaries = [];
      for (int j = 0; j <= currentLevelData.length - grainSize; j += grainSize) {
        List<Tensor<Vector>> chunk = currentLevelData.sublist(j, j + grainSize);
        Tensor<Vector> flattened = concatenateAll(chunk);
        Tensor<Vector> summary = vectorTanh(addVector(matVecMul(aggW[i], flattened), aggB[i]));
        nextLevelSummaries.add(summary);
      }
      tierInputs[i + 1] = nextLevelSummaries;
      currentLevelData = nextLevelSummaries;
    }

    Tensor<Vector> context = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (int i = numTiers - 1; i >= 0; i--) {
      List<Tensor<Vector>> sequenceForThisTier = tierInputs[i];
      int lookback = (i == 0) ? finalSequenceLength : 10;
      List<Tensor<Vector>> finalSequence = sequenceForThisTier.sublist(
          max(0, sequenceForThisTier.length - lookback)
      );

      context = _lstmLoop(
        finalSequence,
        context,
        i,
      );
    }

    return context;
  }

  Tensor<Vector> _lstmLoop(
      List<Tensor<Vector>> sequence,
      Tensor<Vector> initialState,
      int tierIndex,
      ) {
    Tensor<Vector> h = initialState;
    Tensor<Vector> c = Tensor<Vector>(List<double>.filled(hiddenSize, 0.0));

    for (Tensor<Vector> x_t in sequence) {
      Tensor<Vector> combined_input = concatenate(h, x_t);

      Tensor<Vector> f_t = sigmoid(addVector(matVecMul(lstmWf[tierIndex], combined_input), lstmBf[tierIndex]));
      Tensor<Vector> i_t = sigmoid(addVector(matVecMul(lstmWi[tierIndex], combined_input), lstmBi[tierIndex]));
      Tensor<Vector> c_tilde_t = vectorTanh(addVector(matVecMul(lstmWc[tierIndex], combined_input), lstmBc[tierIndex]));

      c = addVector(elementWiseMultiply(f_t, c), elementWiseMultiply(i_t, c_tilde_t));

      Tensor<Vector> o_t = sigmoid(addVector(matVecMul(lstmWo[tierIndex], combined_input), lstmBo[tierIndex]));
      h = elementWiseMultiply(o_t, vectorTanh(c));
    }
    return h;
  }

  Tensor<Vector> concatenateAll(List<Tensor<Vector>> tensors) {
    if (tensors.isEmpty) return Tensor<Vector>([]);
    if (tensors.length == 1) return tensors[0];
    Tensor<Vector> result = tensors[0];
    for (int i = 1; i < tensors.length; i++) {
      result = concatenate(result, tensors[i]);
    }
    return result;
  }

  @override
  Map<String, dynamic> getWeights() {
    Map<String, dynamic> weights = {};

    List<Matrix> serializeMatrixList(List<Tensor<Matrix>> tensorList) {
      List<Matrix> values = [];
      for (Tensor<Matrix> t in tensorList) {
        values.add(t.value);
      }
      return values;
    }

    List<Vector> serializeVectorList(List<Tensor<Vector>> tensorList) {
      List<Vector> values = [];
      for (Tensor<Vector> t in tensorList) {
        values.add(t.value);
      }
      return values;
    }

    weights['lstmWf'] = serializeMatrixList(lstmWf);
    weights['lstmWi'] = serializeMatrixList(lstmWi);
    weights['lstmWc'] = serializeMatrixList(lstmWc);
    weights['lstmWo'] = serializeMatrixList(lstmWo);
    weights['lstmBf'] = serializeVectorList(lstmBf);
    weights['lstmBi'] = serializeVectorList(lstmBi);
    weights['lstmBc'] = serializeVectorList(lstmBc);
    weights['lstmBo'] = serializeVectorList(lstmBo);

    weights['aggW'] = serializeMatrixList(aggW);
    weights['aggB'] = serializeVectorList(aggB);

    return weights;
  }

  @override
  void setWeights(Map<String, dynamic> weightsMap) {

    void _copyMatrixList(List<Tensor<Matrix>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i++) {
        List<dynamic> newMatrixDynamic = newDataList[i] as List<dynamic>;
        Matrix newMatrix = newMatrixDynamic.map((dynamic row) {
          return (row as List<dynamic>).map((dynamic val) => val as double).toList();
        }).toList();

        Tensor<Matrix> tensor = tensorList[i];
        for (int r = 0; r < tensor.value.length; r++) {
          for (int c = 0; c < tensor.value[r].length; c++) {
            tensor.value[r][c] = newMatrix[r][c];
          }
        }
      }
    }

    void _copyVectorList(List<Tensor<Vector>> tensorList, List<dynamic> newDataList) {
      for (int i = 0; i < tensorList.length; i++) {
        List<dynamic> newVector = newDataList[i] as List<dynamic>;
        Tensor<Vector> tensor = tensorList[i];
        for (int j = 0; j < tensor.value.length; j++) {
          tensor.value[j] = newVector[j] as double;
        }
      }
    }

    _copyMatrixList(lstmWf, weightsMap['lstmWf'] as List<dynamic>);
    _copyMatrixList(lstmWi, weightsMap['lstmWi'] as List<dynamic>);
    _copyMatrixList(lstmWc, weightsMap['lstmWc'] as List<dynamic>);
    _copyMatrixList(lstmWo, weightsMap['lstmWo'] as List<dynamic>);
    _copyVectorList(lstmBf, weightsMap['lstmBf'] as List<dynamic>);
    _copyVectorList(lstmBi, weightsMap['lstmBi'] as List<dynamic>);
    _copyVectorList(lstmBc, weightsMap['lstmBc'] as List<dynamic>);
    _copyVectorList(lstmBo, weightsMap['lstmBo'] as List<dynamic>);

    _copyMatrixList(aggW, weightsMap['aggW'] as List<dynamic>);
    _copyVectorList(aggB, weightsMap['aggB'] as List<dynamic>);
  }
}

class ReshapeVectorToMatrixLayer extends Layer {
  @override
  String name = 'reshape_vec_to_mat';
  @override
  List<Tensor> get parameters => [];

  @override
  Tensor<Matrix> forward(Tensor<dynamic> input) {
    Vector v = (input as Tensor<Vector>).value;
    return reshapeVectorToMatrix(input as Tensor<Vector>, 1, v.length);
  }

  @override
  Map<String, dynamic> getWeights() {
    return {};
  }

  @override
  void setWeights(Map<String, dynamic> weights) {
  }
}

void setTrainingMode(SNetwork model, bool isTraining) {
  for (Layer layer in model.layers) {
    if (layer is DropoutLayer) {
      layer.isTraining = isTraining;
    }
    if (layer is DropoutLayerMatrix) {
      layer.isTraining = isTraining;
    }
  }
}