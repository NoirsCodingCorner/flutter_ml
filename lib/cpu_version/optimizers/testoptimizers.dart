
/*import 'optimizer.dart';
import 'sgdmomentum.dart';

import '../../tensor/tensor.dart';
import 'adagrad.dart';
import 'adam.dart';
import 'adamw.dart';
import 'amsgrad.dart';
import 'nag.dart';
import 'rmsprop.dart';

void main() {
  print('--- Starting Optimizer Verification Suite ---');

  // List of optimizer constructors to test
  // We use a simple function to reset the environment for each one
  runTest('Momentum', (List<Tensor> params) {
    return Momentum(params, learningRate: 0.1, momentum: 0.9);
  });

  runTest('NAG (Nesterov)', (List<Tensor> params) {
    return NAG(params, learningRate: 0.1, momentum: 0.9);
  });

  runTest('RMSprop', (List<Tensor> params) {
    return RMSprop(params, learningRate: 0.1);
  });

  runTest('Adagrad', (List<Tensor> params) {
    return Adagrad(params, learningRate: 0.5);
  });

  runTest('Adam', (List<Tensor> params) {
    return Adam(params, learningRate: 0.5);
  });

  runTest('AdamW', (List<Tensor> params) {
    return AdamW(params, learningRate: 0.5, weightDecay: 0.01);
  });

  runTest('AMSGrad', (List<Tensor> params) {
    return AMSGrad(params, learningRate: 0.5);
  });

  print('\n--- All tests completed ---');
}

void runTest(String name, Optimizer Function(List<Tensor>) factory) {
  // Initialize a weight at 0.0 and a target at 10.0
  Tensor<double> weight = Tensor<double>(1.0);
  Tensor<double> target = Tensor<double>(10.0);

  Optimizer optimizer = factory([weight]);

  double initialValue = weight.data[0];

  // Run 5 iterations of optimization
  for (int i = 0; i < 5; i = i + 1) {
    // loss = (weight - target)^2
    // Using simple subtraction and multiplication for a scalar test
    Tensor<double> diff = subtractScalars(weight, target);
    Tensor<double> loss = multiplyScalars(diff, diff);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();
  }

  double finalValue = weight.data[0];
  bool success = (finalValue - initialValue).abs() > 0;
  bool directionCorrect = (finalValue > initialValue);

  if (success && directionCorrect) {
    print('[PASS] $name: Weight moved from ${initialValue.toStringAsFixed(2)} to ${finalValue.toStringAsFixed(2)}');
  } else {
    print('[FAIL] $name: Weight stayed at ${initialValue.toStringAsFixed(2)} or moved wrong way');
  }
}

// Simple helper math for the test logic
Tensor<double> subtractScalars(Tensor<double> a, Tensor<double> b) {
  Tensor<double> out = Tensor<double>(a.data[0] - b.data[0]);
  out.creator = Node(
    [a, b],
        () {
      a.grad[0] = a.grad[0] + out.grad[0];
      b.grad[0] = b.grad[0] - out.grad[0];
    },
    opName: 'sub',
  );
  return out;
}

Tensor<double> multiplyScalars(Tensor<double> a, Tensor<double> b) {
  Tensor<double> out = Tensor<double>(a.data[0] * b.data[0]);
  out.creator = Node(
    [a, b],
        () {
      a.grad[0] = a.grad[0] + out.grad[0] * b.data[0];
      b.grad[0] = b.grad[0] + out.grad[0] * a.data[0];
    },
    opName: 'mul',
  );
  return out;
}*/