import 'dart:math';

import 'Operation.dart';

// Type Aliases
typedef Scalar = double;
typedef Vector = List<double>;
typedef Matrix = List<List<double>>;

class Node {
  final List<Tensor> inputs;
  final Operation op;
  Node(this.inputs, {required this.op});
}

class Tensor {
  final Graph graph;
  final Node? creator;
  final int id;
  Tensor(this.graph, this.id, {this.creator});
}

class Graph {
  int _tensorIdCounter = 0;
  List<Node> nodes = [];
  List<Tensor> parameters = [];
  List<Tensor> tensors = [];
  Map<Tensor, dynamic> initialValues = {};

  Tensor _newTensor({Node? creator}) {
    Tensor t = Tensor(this, _tensorIdCounter++, creator: creator);
    tensors.add(t);
    return t;
  }

  Tensor parameter(dynamic initialValue) {
    Tensor t = _newTensor();
    parameters.add(t);
    initialValues[t] = initialValue;
    return t;
  }

  Tensor constant(dynamic value) {
    Tensor t = _newTensor();
    initialValues[t] = value;
    return t;
  }

  Tensor apply(Operation op, List<Tensor> inputs) {
    Node node = Node(inputs, op: op);
    nodes.add(node);
    return _newTensor(creator: node);
  }
}

class Executor {
  Map<Tensor, dynamic> forward(Graph graph, List<Tensor> outputs) {
    Map<Tensor, dynamic> cache = {};
    for (Tensor t in outputs) {
      _getValue(t, cache);
    }
    Map<Tensor, dynamic> results = {};
    for (Tensor t in outputs) {
      results[t] = cache[t];
    }
    return results;
  }

  dynamic _getValue(Tensor t, Map<Tensor, dynamic> cache) {
    if (cache.containsKey(t)) {
      return cache[t];
    }
    if (t.creator == null) {
      cache[t] = t.graph.initialValues[t];
      return cache[t];
    }
    List<dynamic> inputValues = [];
    for (Tensor inputTensor in t.creator!.inputs) {
      inputValues.add(_getValue(inputTensor, cache));
    }
    dynamic result = t.creator!.op.forward(inputValues);
    cache[t] = result;
    return result;
  }

// Backward pass implementation would follow a similar generic pattern
}

void main() {
  Graph g = Graph();

  Tensor x = g.constant([
    [1.0, 2.0, 3.0],
  ]);

  Tensor W = g.parameter([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
  ]);

  Tensor b = g.parameter([10.0, 20.0]);

  Tensor y_linear = g.apply(MatMulOp(), [x, W]);
  Tensor y = g.apply(AddMatrixAndVectorOp(), [y_linear, b]);

  print('--- Graph Definition Complete ---');
  print('Graph has ${g.nodes.length} operations defined.');

  Executor ex = Executor();
  Map<Tensor, dynamic> results = ex.forward(g, [y]);

  print('\n--- Execution Complete ---');
  print('Final output value: ${results[y]}');
}