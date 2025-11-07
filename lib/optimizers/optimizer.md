
## `Optimizer` Class

The `Optimizer` class is the abstract base class for all optimization algorithms.

An `Optimizer`'s role is to hold a model's trainable `parameters` and update them according to a specific algorithm (e.g., `SGD`, `Adam`) using the gradients computed during the backward pass.

This abstraction allows the training loop to remain generic; you can easily swap out one optimizer for another without changing the training code.

### Training Workflow

The typical training loop sequence is:

1.  **`loss.backward()`** - Compute gradients for all parameters.
2.  **`optimizer.step()`** - Update all parameters using the gradients.
3.  **`optimizer.zeroGrad()`** - Reset all gradients to zero for the next iteration.

### Example

This example shows how to manually use an optimizer with an `SNetwork`.

```dart
// 1. Define a model (e.g., using SNetwork)
SNetwork myModel = SNetwork([
  DenseLayer(8),
  ReLULayer(),
  DenseLayer(1)
]);

// 2. Build the model to initialize parameters
// (A dummy input is fine for this)
myModel.predict(Tensor<Vector>([0.0, 0.0]));

// 3. Create an optimizer with the model's parameters
Optimizer optimizer = Adam(myModel.parameters, learningRate: 0.001);

// 4. Inside the training loop (showing one step):
Tensor<Vector> input = Tensor<Vector>([1.0, 2.0]);
Tensor<Vector> target = Tensor<Vector>([1.0]);

// --- The core optimizer workflow ---
Tensor<Vector> output = myModel.forward(input) as Tensor<Vector>; // 1. Get prediction
Tensor<Scalar> loss = mse(output, target); // 2. Calculate loss
loss.backward();                          // 3. Compute gradients
optimizer.step();                         // 4. Update parameters
optimizer.zeroGrad();                     // 5. Reset gradients
// --- End of training step ---
```

-----

### API Reference

* **`Optimizer(List<Tensor> parameters, {required double learningRate})`**

    * The base constructor. It requires the list of model `parameters` (e.l., `myModel.parameters`) and a `learningRate`.

* **`void step()`**

    * (Abstract) Performs a single optimization step (parameter update). Each optimizer implements its own unique update rule.

* **`void zeroGrad()`**

    * Resets the gradients of all parameters to zero. This must be called at the end of each training iteration.

-----

## Available Optimizers

This file exports the following optimizer implementations:

* **`SGD`**: The standard, "vanilla" Stochastic Gradient Descent.
* **`Momentum`**: `SGD` with the addition of a momentum (velocity) term.
* **`NAG`**: Nesterov Accelerated Gradient, an improvement on `Momentum`.
* **`Adagrad`**: An adaptive optimizer good for sparse data.
* **`RMSprop`**: An adaptive optimizer that performs well with RNNs.
* **`Adam`**: The most common, general-purpose adaptive optimizer.
* **`AMSGrad`**: A variant of `Adam` that fixes a potential convergence issue.
* **`AdamW`**: A variant of `Adam` that improves weight decay (L2 regularization).

-----

## Optimizer Guide

### `SGD` (Stochastic Gradient Descent)

* **Description:** The most fundamental algorithm. It updates parameters using the simple rule: `param = param - lr * grad`.
* **Use Case:** A simple, reliable baseline. While often slower than other optimizers, it can lead to good model generalization.

### `Momentum`

* **Description:** An extension of `SGD` that adds a "velocity" (a moving average of past gradients). This helps accelerate in consistent directions and dampen oscillations.
* **Use Case:** Often converges much faster than standard `SGD`, especially in deep networks or on noisy data.

### `NAG` (Nesterov Accelerated Gradient)

* **Description:** An improvement on `Momentum` that "looks ahead." It calculates the gradient *after* making a preliminary step in the velocity direction, which helps prevent overshooting.
* **Use Case:** A refinement of `Momentum` that often provides faster and more stable convergence.

### `Adagrad`

* **Description:** An adaptive learning rate optimizer. It gives each parameter a unique learning rate by dividing by the sum of all past squared gradients.
* **Use Case:** Excellent for sparse data (like NLP word embeddings), where some parameters are updated very infrequently.

### `RMSprop` (Root Mean Square Propagation)

* **Description:** Another adaptive optimizer. It fixes `Adagrad`'s aggressive learning rate decay by using a moving average of squared gradients instead of the full sum.
* **Use Case:** A very effective and popular optimizer, especially for Recurrent Neural Networks (RNNs).

### `Adam` (Adaptive Moment Estimation)

* **Description:** The most common adaptive optimizer. It combines the ideas of `Momentum` (tracking the first-moment estimate) and `RMSprop` (tracking the second-moment estimate).
* **Use Case:** The default, go-to optimizer for most problems. It is fast, efficient, and generally works well with default settings.

### `AMSGrad`

* **Description:** A variant of `Adam` that fixes a potential convergence issue by using the maximum of past squared gradients for normalization, ensuring the adaptive learning rate is non-increasing.
* **Use Case:** Use as a replacement for `Adam` if you are experiencing stability or convergence issues on a specific problem.

### `AdamW`

* **Description:** A popular `Adam` variant that decouples weight decay (L2 regularization) from the adaptive learning rate update. This separation often leads to better model generalization.
* Setting `weightDecay` in `Adam` is not the same as `AdamW`.
* **Use Case:** The default choice for modern, large models like Transformers. Use this if you are applying significant weight decay.