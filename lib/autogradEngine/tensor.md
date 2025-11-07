## `Tensor` Mathematics

The library's autograd engine is built upon two fundamental classes: `Tensor` and `Node`. The `Tensor` class manages data, while the `Node` class represents the operations that connect tensors.

-----

### The `Tensor` Class

The `Tensor` is the primary data structure, acting as a container for numerical buffers.

**Supported Data Types:**
The generic type `T` of a `Tensor<T>` can be one of the following:

* `Scalar` (a `double`)
* `Vector` (a `List<double>`)
* `Matrix` (a `List<List<double>>`)
* `Tensor3D` (a `List<List<List<double>>>`)

**Properties:**
A `Tensor` instance holds three key properties:

* **`value`**: The numerical data of the tensor.
* **`grad`**: The gradient of the tensor, which has the same shape as `value`.
* **`creator`**: An optional `Node` object that references the operation that produced this tensor. If `creator` is `null`, the tensor is considered a "leaf" (a user-created input).

**Constructor:**
When a `Tensor` is initialized, its `grad` property is automatically created and filled with zeros, matching the shape of the `value`.

```dart
  Tensor(this.value, {Node? creator2}) {
    creator = creator2;
    if (value is Scalar) {
      grad = 0.0 as T;
    } else if (value is Vector) {
      Vector valAsList = value as Vector;
      grad = List<double>.filled(valAsList.length, 0.0) as T;
    } else if (value is Matrix) {
      Matrix valAsMatrix = value as Matrix;
      int numRows = valAsMatrix.length;
      int numCols = valAsMatrix.isNotEmpty ? valAsMatrix[0].length : 0;
      grad =
          List.generate(numRows, (_) => List<double>.filled(numCols, 0.0)) as T;
    }
  }
```

**Methods:**
Key methods for the autograd engine include:

* `backward()`: Initiates the backward pass to compute gradients.
* `zeroGrad()`: Resets the gradients of this tensor and its antecedents.
* `printGraph()`: Visualizes the computational graph.
* `printParallelGraph()`: Visualizes the graph grouped by parallelizable levels.

-----

### The `Node` Class

The `Node` class represents a single operation within the computational graph. It stores the context required to compute gradients.

**Properties:**

* **`inputs`**: A `List<Tensor>` containing the tensors used as input for this operation.
* **`backwardFn`**: A `Function` (closure) that defines the gradient calculation (the chain rule) specific to this operation.
* **`opName`**: A string name for the operation (e.g., 'add', 'matMul').
* **`cost`**: An estimated computational cost for the operation.
* **`extraParams`**: A `Map` to store any non-tensor parameters (like constants or shape information) needed by the `backwardFn`.

-----

### How It Works: The Forward Pass

The "forward pass" is the process of executing operations to compute an output value. During this pass, the computational graph is constructed.

Consider the following operation:

```dart
Tensor a = Tensor(5.0);
Tensor b = Tensor(10.0);
Tensor c = add(a, b);
```

The `add` function (for scalars) performs the following steps:

```dart
Tensor<Scalar> add(Tensor<Scalar> a, Tensor<Scalar> b) {
  Scalar outValue = a.value + b.value;
  Tensor<Scalar> out = Tensor<Scalar>(outValue);
  out.creator = Node([a, b], () {
    /* backwardFn */
    a.grad += out.grad;
    b.grad += out.grad;
  }, opName: 'add');
  return out;
}
```

1.  **Compute Value**: It calculates the forward result, `outValue = a.value + b.value`.
2.  **Create Output Tensor**: It initializes a new tensor, `out`, with this `outValue`.
3.  **Build Graph**: It creates a new `Node` and assigns it to `out.creator`. This `Node` "remembers" that its `inputs` were `[a, b]` and stores the `backwardFn` required to compute their gradients.

The result is `c`, a new tensor containing the value `15.0`. Its `creator` property now links to the `Node` that connects it back to `a` and `b`.

-----

### How It Works: The Backward Pass

The "backward pass" computes the gradients for all tensors in the graph, starting from an output tensor. It is initiated by calling `backward()`.

```dart
c.backward();
```

The `backward()` method executes the following logic:

1.  **Topological Sort**: It first traverses the graph backward from the current tensor (e.g., `c`) via the `creator` links. It builds a `topo` list of all `Node` objects in the correct order of dependency.
2.  **Initialize Gradient**: It sets the starting gradient for the tensor `backward()` was called on (e.g., `c.grad`) to `1.0`.
3.  **Propagate Gradients**: It iterates through the `topo` list in reverse. For each `Node`, it executes its stored `backwardFn`.

In our example, the `backwardFn` from the `add` operation is called:

```dart
() {
  a.grad += out.grad; // out is c
  b.grad += out.grad; // out is c
}
```

This function accesses the gradient of its output (`c.grad`, which is `1.0`) and propagates it to its inputs (`a` and `b`), summing it into their respective `.grad` properties. If `a` or `b` had their own `creator` nodes, the process would continue until all leaf nodes are reached.

Here is a reference list of the methods and operations defined in the `tensor.dart` file.

## Type Aliases

These types for clarity.

* `typedef Scalar = double;`
* `typedef Vector = List<double>;`
* `typedef Matrix = List<List<double>>;`
* `typedef Tensor3D = List<List<List<double>>>;`

---

## Scalar (0D) Operations

* `Tensor<Scalar> add(Tensor<Scalar> a, Tensor<Scalar> b)`
* `Tensor<Scalar> multiplyScalar(Tensor<Scalar> a, Tensor<Scalar> b)`
* `Tensor<Scalar> multiply(Tensor<Scalar> a, Tensor<Scalar> b)`
* `Tensor<Scalar> sigmoidScalar(Tensor<Scalar> s)`
* `Tensor<Scalar> binaryCrossEntropy(Tensor<Scalar> prediction, Tensor<Scalar> target)`

---

## Vector (1D) Operations

* `Tensor<Vector> addVector(Tensor<Vector> a, Tensor<Vector> b)`
* `Tensor<Vector> addScalar(Tensor<Vector> v, double s)`
* `Tensor<Vector> concatenate(Tensor<Vector> a, Tensor<Vector> b)`
* `Tensor<Scalar> dot(Tensor<Vector> a, Tensor<Vector> b)`
* `Tensor<Vector> elementWiseMultiply(Tensor<Vector> a, Tensor<Vector> b)`
* `Tensor<Scalar> mse(Tensor<Vector> predictions, Tensor<Vector> targets)`
* `Tensor<Vector> relu(Tensor<Vector> v)`
* `Tensor<Vector> sigmoid(Tensor<Vector> v)`
* `Tensor<Scalar> sum(Tensor<Vector> v)`
* `Tensor<Vector> vectorTanh(Tensor<Vector> v)`
* `Tensor<Vector> vectorExp(Tensor<Vector> v)`
* `Tensor<Vector> vectorLog(Tensor<Vector> v)`

---

## Matrix (2D) Operations

* `Tensor<Matrix> addMatrix(Tensor<Matrix> a, Tensor<Matrix> b)`
* `Tensor<Matrix> addMatrixAndVector(Tensor<Matrix> m, Tensor<Vector> v)`
* `Tensor<Matrix> addScalarToMatrix(Tensor<Matrix> m, Tensor<Scalar> s)`
* `Tensor<Matrix> concatenateMatricesByColumn(List<Tensor<Matrix>> matrices)`
* `Tensor<Matrix> conv2d(Tensor<Matrix> input, Tensor<Matrix> kernel, {String padding = 'valid'})`
* `Tensor<Matrix> elementWiseMultiplyMatrix(Tensor<Matrix> a, Tensor<Matrix> b)`
* `Tensor<Matrix> matMul(Tensor<Matrix> a, Tensor<Matrix> b)`
* `Tensor<Vector> matVecMul(Tensor<Matrix> M, Tensor<Vector> v)`
* `Tensor<Scalar> mseMatrix(Tensor<Matrix> predictions, Tensor<Matrix> targets)`
* `Tensor<Matrix> reluMatrix(Tensor<Matrix> m)`
* `Tensor<Matrix> reshapeVectorToMatrix(Tensor<Vector> v, int numRows, int numCols)`
* `Tensor<Matrix> scaleMatrix(Tensor<Matrix> m, double s)`
* `Tensor<Vector> selectRow(Tensor<Matrix> m, int rowIndex)`
* `Tensor<Matrix> sigmoidMatrix(Tensor<Matrix> m)`
* `Tensor<Scalar> sumMatrix(Tensor<Matrix> m)`
* `Tensor<Matrix> tanhMatrix(Tensor<Matrix> m)`
* `Tensor<Matrix> transpose(Tensor<Matrix> a)`

---

## 3D Tensor Operations

* `Tensor<Tensor3D> add3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b)`
* `Tensor<Tensor3D> elementWiseMultiply3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b)`
* `Tensor<Tensor3D> concatenate3D(Tensor<Tensor3D> a, Tensor<Tensor3D> b)`

---

## Composite Operations

These are functions composed of the primitive operations listed above.

* `Tensor<Vector> softplus(Tensor<Vector> v)`

---

## Utility Functions (No Graph)

These functions operate directly on data types, not `Tensor` objects, and do not create graph nodes.

* `Matrix padMatrix(Matrix input, int padding)`
