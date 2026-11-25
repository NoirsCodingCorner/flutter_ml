import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_ml/full_library.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  setUp(() {
    Logger.cyan(
      'Resetting TensorErrorHandler listeners and error log...',
      prefix: '🔄',
    );
    TensorErrorHandler.clearListeners();
    TensorErrorHandler.clearLog();
  });

  test('reacts to shape mismatch in matrix multiplication', () {
    Logger.blue('--- Test 1: Shape Mismatch in matMul ---', prefix: '🧪');
    List<String> alerts = [];

    Logger.cyan(
      'Registering sizeMismatch listener "shape_monitor"...',
      prefix: '👂',
    );
    TensorErrorHandler.on(
      TensorErrorType.sizeMismatch,
      'shape_monitor',
          (TensorError error) {
        String alertMsg = '${error.opName}: ${error.message}';
        alerts.add(alertMsg);
        Logger.yellow('Listener triggered -> $alertMsg', prefix: '⚠️');
      },
    );

    Tensor<Matrix> a = Tensor<Matrix>([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
    Tensor<Matrix> b = Tensor<Matrix>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    Logger.cyan(
      'Created Matrix ${a.id} ${a.shape} and Matrix ${b.id} ${b.shape}. Attempting matMul...',
      prefix: '📐',
    );

    bool caught = false;
    try {
      matMul(a, b);
    } on TensorException catch (e) {
      caught = true;
      Logger.red('Caught ${e.toString()}', prefix: '🛑');
      expect(e.error.type, TensorErrorType.sizeMismatch);
      expect(e.error.opName, 'matMul');
      expect(e.error.tensors.length, 2);
    }

    expect(caught, true);
    expect(alerts.length, 1);
    expect(TensorErrorHandler.errorLog.containsKey('err_0'), true);
    Logger.green(
      'Verified errorLog entry "err_0" and freed tensors.',
      prefix: '✅',
    );

    a.free();
    b.free();
  });

  test('reacts to out-of-range prediction in binary cross entropy', () {
    Logger.blue(
      '--- Test 2: Out-of-Range Prediction in binaryCrossEntropy ---',
      prefix: '🧪',
    );
    Tensor<Scalar> prediction = Tensor<Scalar>(1.0);
    Tensor<Scalar> target = Tensor<Scalar>(1.0);
    Logger.cyan(
      'Created prediction (${prediction.value}) and target (${target.value}). Running binaryCrossEntropy...',
      prefix: '📊',
    );

    bool fallbackUsed = false;
    Tensor<Scalar> loss;

    try {
      loss = binaryCrossEntropy(prediction, target);
    } on TensorException catch (e) {
      Logger.red('Caught ${e.toString()}', prefix: '🛑');
      expect(e.error.type, TensorErrorType.mathError);
      expect(e.error.extraParams['value'], 1.0);

      Logger.yellow(
        'Invalid value ${e.error.extraParams['value']} detected. Clamping prediction to 0.9999 and retrying...',
        prefix: '🔧',
      );
      prediction.data[0] = 0.9999;
      loss = binaryCrossEntropy(prediction, target);
      fallbackUsed = true;
    }

    expect(fallbackUsed, true);
    expect(loss.value, closeTo(0.0001, 0.0001));
    Logger.green(
      'Retry succeeded! Computed loss: ${loss.value}',
      prefix: '✅',
    );

    prediction.free();
    target.free();
    loss.free();
  });

  test('reacts to invalid dropout rate parameter', () {
    Logger.blue(
      '--- Test 3: Invalid Dropout Rate Parameter ---',
      prefix: '🧪',
    );
    Tensor<Vector> activations = Tensor<Vector>([0.5, 1.5, -0.5]);
    Logger.cyan(
      'Created activations ${activations.id} ${activations.shape}. Calling dropoutVectorMath with rate = 1.0...',
      prefix: '🎲',
    );

    bool caught = false;
    try {
      dropoutVectorMath(activations, 1.0, true);
    } on TensorException catch (e) {
      caught = true;
      Logger.red('Caught ${e.toString()}', prefix: '🛑');
      Logger.yellow(
        'Rejected rate parameter: ${e.error.extraParams['rate']}',
        prefix: '⚠️',
      );
      expect(e.error.type, TensorErrorType.invalidParameter);
      expect(e.error.extraParams['rate'], 1.0);
    }

    expect(caught, true);
    Logger.green(
      'Invalid dropout rate properly rejected and freed tensors.',
      prefix: '✅',
    );

    activations.free();
  });

  test('reacts to operations on a freed tensor', () {
    Logger.blue(
      '--- Test 4: Operations on a Freed Tensor ---',
      prefix: '🧪',
    );
    Tensor<Vector> v1 = Tensor<Vector>([1.0, 2.0]);
    Tensor<Vector> v2 = Tensor<Vector>([3.0, 4.0]);
    Logger.cyan(
      'Created vectors ${v1.id} and ${v2.id}.',
      prefix: '📦',
    );

    v1.free();
    Logger.yellow(
      'Freed ${v1.id} (isFreed: ${v1.isFreed}). Attempting addVector(${v1.id}, ${v2.id})...',
      prefix: '🗑️',
    );

    bool detectedFreed = false;
    try {
      addVector(v1, v2);
    } on TensorException catch (e) {
      Logger.red('Caught ${e.toString()}', prefix: '🛑');
      detectedFreed = e.error.type == TensorErrorType.emptyData;
    }

    expect(detectedFreed, true);
    Logger.green(
      'Verified emptyData protection on freed tensor and freed ${v2.id}.',
      prefix: '✅',
    );

    v2.free();
  });
}