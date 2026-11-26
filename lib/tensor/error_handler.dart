import 'package:flutter_ml/full_library.dart';

enum TensorErrorType {
  sizeMismatch,
  mathError,
  invalidParameter,
  emptyData,
}

typedef TensorErrorCallback = void Function(TensorError error);

class TensorError {
  TensorErrorType type;
  String opName;
  String message;
  List<Tensor> tensors;
  Map<String, dynamic> extraParams;

  TensorError({
    required this.type,
    required this.opName,
    required this.message,
    this.tensors = const [],
    this.extraParams = const {},
  });

  @override
  String toString() {
    String tensorInfo = '';
    for (int i = 0; i < tensors.length; i = i + 1) {
      tensorInfo = '$tensorInfo${tensors[i].id} ${tensors[i].shape}';
      if (i < tensors.length - 1) {
        tensorInfo = '$tensorInfo, ';
      }
    }

    if (tensorInfo.isEmpty) {
      return '[$opName] (${type.name}): $message';
    }
    return '[$opName] (${type.name}): $message | Tensors: [$tensorInfo]';
  }
}

class TensorException implements Exception {
  TensorError error;

  TensorException(this.error);

  @override
  String toString() {
    return 'TensorException: ${error.toString()}';
  }
}

class TensorErrorHandler {
  static int bufferSize = 50;
  static int _errorCounter = 0;

  static Map<TensorErrorType, Map<String, TensorErrorCallback>> _listeners = {};
  static Map<String, TensorError> errorLog = {};

  /// Registers or overwrites a named [callback] for a specific [type].
  static void on(
      TensorErrorType type,
      String listenerName,
      TensorErrorCallback callback,
      ) {
    if (!_listeners.containsKey(type)) {
      _listeners[type] = {};
    }
    _listeners[type]![listenerName] = callback;
  }

  /// Removes a specific named listener from [type].
  static void off(TensorErrorType type, String listenerName) {
    if (_listeners.containsKey(type)) {
      _listeners[type]!.remove(listenerName);
    }
  }

  /// Clears listeners for [type], or all listeners if [type] is omitted.
  static void clearListeners({TensorErrorType? type}) {
    if (type != null) {
      _listeners.remove(type);
    } else {
      _listeners.clear();
    }
  }

  /// Clears the recorded error trace buffer.
  static void clearLog() {
    errorLog.clear();
    _errorCounter = 0;
  }

  /// Records [error] into [errorLog], triggers registered callbacks for [error.type],
  /// and throws a [TensorException].
  static Never report(TensorError error) {
    if (bufferSize > 0) {
      while (errorLog.length >= bufferSize) {
        String oldestKey = errorLog.keys.first;
        errorLog.remove(oldestKey);
      }
      String errorId = 'err_$_errorCounter';
      _errorCounter = _errorCounter + 1;
      errorLog[errorId] = error;
    }

    Map<String, TensorErrorCallback>? callbacks = _listeners[error.type];
    if (callbacks != null) {
      List<TensorErrorCallback> callbackList = callbacks.values.toList();
      for (int i = 0; i < callbackList.length; i = i + 1) {
        callbackList[i](error);
      }
    }

    throw TensorException(error);
  }


}

class TensorValidator {
  // ─────────────────────────────────────────────────────── //
  // DATA & LIFECYCLE CHECKS (emptyData)
  // ─────────────────────────────────────────────────────── //

  /// Verifies that [t] has not been freed and contains at least one element.
  static void requireValidData(Tensor t, String opName) {
    if (t.isFreed || t.data.isEmpty) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.emptyData,
          opName: opName,
          message: '${t.id} is freed or contains no data.',
          tensors: [t],
        ),
      );
    }
  }

  // ─────────────────────────────────────────────────────── //
  // SHAPE & DIMENSION CHECKS (sizeMismatch)
  // ─────────────────────────────────────────────────────── //

  /// Verifies that [a] and [b] have identical ranks and dimensions.
  /// Used for: addVector, dot, elementWiseMultiply, mse, addMatrix,
  /// elementWiseMultiplyMatrix, mseMatrix, add3D, elementWiseMultiply3D.
  static void requireSameShape(Tensor a, Tensor b, String opName) {
    requireValidData(a, opName);
    requireValidData(b, opName);

    bool mismatch = a.shape.length != b.shape.length;
    if (!mismatch) {
      for (int i = 0; i < a.shape.length; i = i + 1) {
        if (a.shape[i] != b.shape[i]) {
          mismatch = true;
          break;
        }
      }
    }

    if (mismatch) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message: '${a.id} ${a.shape} and ${b.id} ${b.shape} do not align.',
          tensors: [a, b],
        ),
      );
    }
  }

  /// Verifies inner dimension alignment for matrix multiplication: [M, N] * [N, P].
  static void requireMatMul(Tensor<Matrix> a, Tensor<Matrix> b, String opName) {
    requireValidData(a, opName);
    requireValidData(b, opName);

    if (a.shape[1] != b.shape[0]) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          'Inner dimensions do not match for matMul: ${a.id} ${a.shape} and ${b.id} ${b.shape}.',
          tensors: [a, b],
        ),
      );
    }
  }

  /// Verifies that matrix [m] columns match vector [v] length: [rows, cols] with [cols].
  /// Used for: matVecMul, addMatrixAndVector.
  static void requireMatrixColsMatchVector(
      Tensor<Matrix> m, Tensor<Vector> v, String opName) {
    requireValidData(m, opName);
    requireValidData(v, opName);

    if (m.shape[1] != v.shape[0]) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          '${m.id} columns (${m.shape[1]}) do not match ${v.id} length (${v.shape[0]}).',
          tensors: [m, v],
        ),
      );
    }
  }

  /// Verifies a list of matrices for stacking or column-wise concatenation.
  /// Set [requireSameCols] to `true` for `stackMatricesTo3D` and `false` for `concatenateMatricesByColumn`.
  static void requireMatrixListCompatible(
      List<Tensor<Matrix>> matrices,
      String opName, {
        bool requireSameCols = true,
      }) {
    if (matrices.isEmpty) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.emptyData,
          opName: opName,
          message: 'Input matrix list is empty.',
        ),
      );
    }

    requireValidData(matrices[0], opName);
    int expectedRows = matrices[0].shape[0];
    int expectedCols = matrices[0].shape[1];

    for (int i = 1; i < matrices.length; i = i + 1) {
      Tensor<Matrix> current = matrices[i];
      requireValidData(current, opName);

      bool rowMismatch = current.shape[0] != expectedRows;
      bool colMismatch = requireSameCols && current.shape[1] != expectedCols;

      if (rowMismatch || colMismatch) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.sizeMismatch,
            opName: opName,
            message:
            '${current.id} ${current.shape} is incompatible with ${matrices[0].id} ${matrices[0].shape}.',
            tensors: [matrices[0], current],
          ),
        );
      }
    }
  }

  /// Verifies that two 3D tensors have matching height and width for axis-0 concatenation.
  static void requireConcat3D(
      Tensor<Tensor3D> a, Tensor<Tensor3D> b, String opName) {
    requireValidData(a, opName);
    requireValidData(b, opName);

    if (a.shape[1] != b.shape[1] || a.shape[2] != b.shape[2]) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          'Spatial dimensions of ${a.id} ${a.shape} and ${b.id} ${b.shape} do not match.',
          tensors: [a, b],
        ),
      );
    }
  }

  /// Verifies that [targetShape] has positive dimensions and the exact same element count as [t].
  static void requireReshape(
      Tensor t, List<int> targetShape, String opName) {
    requireValidData(t, opName);

    int targetCount = 1;
    for (int i = 0; i < targetShape.length; i = i + 1) {
      if (targetShape[i] <= 0) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.invalidParameter,
            opName: opName,
            message: 'Target shape dimensions must be > 0, got $targetShape.',
            tensors: [t],
            extraParams: {'targetShape': targetShape},
          ),
        );
      }
      targetCount = targetCount * targetShape[i];
    }

    if (t.data.length != targetCount) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          'Cannot reshape ${t.id} (${t.data.length} elements) into $targetShape ($targetCount elements).',
          tensors: [t],
          extraParams: {'targetShape': targetShape},
        ),
      );
    }
  }

  // ─────────────────────────────────────────────────────── //
  // PARAMETER & WINDOW CHECKS (invalidParameter / sizeMismatch)
  // ─────────────────────────────────────────────────────── //

  /// Verifies that [index] is within `[0, maxExclusive)`. Used for `selectRow`.
  static void requireIndex(
      int index, int maxExclusive, Tensor t, String opName) {
    requireValidData(t, opName);

    if (index < 0 || index >= maxExclusive) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message:
          'Index $index out of bounds [0, ${maxExclusive - 1}] for ${t.id} ${t.shape}.',
          tensors: [t],
          extraParams: {'index': index},
        ),
      );
    }
  }

  /// Verifies 1D pooling window size, stride, and input length.
  static void requirePool1D(
      Tensor<Vector> input, int poolSize, int stride, String opName) {
    requireValidData(input, opName);

    if (poolSize <= 0 || stride <= 0) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message:
          'poolSize ($poolSize) and stride ($stride) must be greater than 0.',
          tensors: [input],
          extraParams: {'poolSize': poolSize, 'stride': stride},
        ),
      );
    }

    if (input.shape[0] < poolSize) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          '${input.id} length (${input.shape[0]}) is smaller than poolSize ($poolSize).',
          tensors: [input],
          extraParams: {'poolSize': poolSize},
        ),
      );
    }
  }

  /// Verifies 2D kernel/window dimensions, stride, padding mode, and input spatial bounds.
  /// Used for: avgPool2d, maxPool2d, conv2d.
  static void requireWindow2D(
      Tensor<Matrix> input,
      int windowHeight,
      int windowWidth,
      int stride,
      String opName, {
        String padding = 'valid',
        Tensor? kernel,
      }) {
    requireValidData(input, opName);
    if (kernel != null) {
      requireValidData(kernel, opName);
    }

    List<Tensor> involved = kernel != null ? [input, kernel] : [input];

    if (padding != 'valid' && padding != 'same') {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message: 'Unsupported padding "$padding". Expected "valid" or "same".',
          tensors: involved,
          extraParams: {'padding': padding},
        ),
      );
    }

    if (windowHeight <= 0 || windowWidth <= 0 || stride <= 0) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message:
          'Window [$windowHeight, $windowWidth] and stride ($stride) must be > 0.',
          tensors: involved,
        ),
      );
    }

    if (padding == 'valid' &&
        (input.shape[0] < windowHeight || input.shape[1] < windowWidth)) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          '${input.id} ${input.shape} is smaller than window [$windowHeight, $windowWidth].',
          tensors: involved,
        ),
      );
    }
  }

  /// Verifies dropout probability rate is within `[0.0, 1.0)` to prevent division by zero.
  static void requireDropoutRate(double rate, Tensor input, String opName) {
    requireValidData(input, opName);

    if (rate < 0.0 || rate >= 1.0) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message: 'Dropout rate must be in [0.0, 1.0), got $rate.',
          tensors: [input],
          extraParams: {'rate': rate},
        ),
      );
    }
  }

  /// Verifies BatchNorm feature/channel alignment and epsilon > 0.
  static void requireBatchNorm(
      Tensor x,
      Tensor<Vector> gamma,
      Tensor<Vector> beta,
      Vector runningMean,
      Vector runningVariance,
      int expectedSize,
      double epsilon,
      String opName,
      ) {
    requireValidData(x, opName);
    requireValidData(gamma, opName);
    requireValidData(beta, opName);

    if (epsilon <= 0.0) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.invalidParameter,
          opName: opName,
          message: 'Epsilon must be > 0.0, got $epsilon.',
          tensors: [x],
          extraParams: {'epsilon': epsilon},
        ),
      );
    }

    if (x.shape[0] != expectedSize ||
        gamma.shape[0] != expectedSize ||
        beta.shape[0] != expectedSize ||
        runningMean.length != expectedSize ||
        runningVariance.length != expectedSize) {
      TensorErrorHandler.report(
        TensorError(
          type: TensorErrorType.sizeMismatch,
          opName: opName,
          message:
          'BatchNorm parameter sizes do not match expected size ($expectedSize).',
          tensors: [x, gamma, beta],
          extraParams: {'expectedSize': expectedSize},
        ),
      );
    }
  }

  // ─────────────────────────────────────────────────────── //
  // NUMERICAL DOMAIN CHECKS (mathError)
  // ─────────────────────────────────────────────────────── //

  /// Verifies all elements in [t] are strictly greater than 0.0 (e.g. for `vectorLog`).
  static void requireStrictlyPositive(Tensor t, String opName) {
    requireValidData(t, opName);

    for (int i = 0; i < t.data.length; i = i + 1) {
      double val = t.data[i];
      if (val <= 0.0 || val.isNaN) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.mathError,
            opName: opName,
            message:
            '${t.id} contains non-positive or NaN value ($val) at flat index $i.',
            tensors: [t],
            extraParams: {'index': i, 'value': val},
          ),
        );
      }
    }
  }

  /// Verifies all elements in [t] lie strictly inside `(0.0, 1.0)` (e.g. for `binaryCrossEntropy`).
  static void requireOpenUnitInterval(Tensor t, String opName) {
    requireValidData(t, opName);

    for (int i = 0; i < t.data.length; i = i + 1) {
      double val = t.data[i];
      if (val <= 0.0 || val >= 1.0 || val.isNaN) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.mathError,
            opName: opName,
            message:
            '${t.id} value ($val) at flat index $i must be strictly between 0.0 and 1.0.',
            tensors: [t],
            extraParams: {'index': i, 'value': val},
          ),
        );
      }
    }
  }

  /// Verifies that no element in [t] is 0.0 or NaN (e.g. for `divide` and `divideTensor`).
  static void requireNonZero(Tensor t, String opName) {
    requireValidData(t, opName);

    for (int i = 0; i < t.data.length; i = i + 1) {
      double val = t.data[i];
      if (val == 0.0 || val.isNaN) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.mathError,
            opName: opName,
            message:
            '${t.id} contains zero or NaN value ($val) at flat index $i.',
            tensors: [t],
            extraParams: {'index': i, 'value': val},
          ),
        );
      }
    }
  }

  /// Verifies all elements in [t] are greater than or equal to 0.0 and not NaN (e.g. for `scalarSqrt`).
  static void requireNonNegative(Tensor t, String opName) {
    requireValidData(t, opName);

    for (int i = 0; i < t.data.length; i = i + 1) {
      double val = t.data[i];
      if (val < 0.0 || val.isNaN) {
        TensorErrorHandler.report(
          TensorError(
            type: TensorErrorType.mathError,
            opName: opName,
            message:
            '${t.id} contains negative or NaN value ($val) at flat index $i.',
            tensors: [t],
            extraParams: {'index': i, 'value': val},
          ),
        );
      }
    }
  }
}
