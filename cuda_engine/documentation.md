# graph_executor

The `GraphExecutor` class is a lightweight execution engine designed to process a serialized sequence of instructions, referred to as a "tape." It reads specific operation codes and their associated data from a byte array and dispatches the corresponding neural network and tensor operations to a `CudaManager` for execution. This setup ensures that a host language (like Dart) can efficiently trigger high-performance GPU operations.

## Methods

* **`GraphExecutor(CudaManager& manager, bool debug = false)`**
  The constructor initializes the executor. It takes a reference to a `CudaManager` instance, which handles the underlying memory and hardware calls, and an optional boolean flag for enabling debug mode.

* **`void run_tape(const uint8_t* tape, int total_bytes, bool sync = true)`**
  The primary execution method. It takes a pointer to the byte array (the tape), the total size of the tape in bytes, and a boolean flag. It reads the sequence of operation codes from the tape and executes them sequentially. If `sync` is true, it forces synchronization of the CUDA stream before the method returns.

## Available Operation Codes

1. `OP_LOAD_SAMPLE`
2. `OP_STORE_SAMPLE`
3. `OP_COPY`
4. `OP_FILL`
5. `OP_ADD`
6. `OP_ADD_INTO`
7. `OP_SUBTRACT`
8. `OP_SUBTRACT_INTO`
9. `OP_MULTIPLY`
10. `OP_MULTIPLY_BACKWARD`
11. `OP_DIVIDE`
12. `OP_DIVIDE_BACKWARD`
13. `OP_MATMUL`
14. `OP_TRANSPOSE`
15. `OP_BROADCAST_ADD`
16. `OP_RELU`
17. `OP_RELU_BACKWARD`
18. `OP_SIGMOID`
19. `OP_SIGMOID_BACKWARD`
20. `OP_TANH`
21. `OP_TANH_BACKWARD`
22. `OP_EXP_ELEMENTWISE`
23. `OP_EXP_BACKWARD`
24. `OP_SOFTMAX_FORWARD`
25. `OP_SOFTMAX_BACKWARD`
26. `OP_MSE_LOSS_FORWARD`
27. `OP_MSE_LOSS_BACKWARD`
28. `OP_BCE_LOSS_FORWARD`
29. `OP_BCE_LOSS_BACKWARD`
30. `OP_MATMUL_BIAS_RELU_FORWARD`
31. `OP_LAYER_NORM_FORWARD`
32. `OP_LAYER_NORM_BACKWARD`
33. `OP_CONV2D_FORWARD`
34. `OP_CONV2D_BACKWARD_INPUT`
35. `OP_CONV2D_BACKWARD_KERNEL`
36. `OP_IM2COL`
37. `OP_COL2IM`
38. `OP_SLICE_ROW`
39. `OP_SLICE_ROW_BACKWARD`
40. `OP_SLICE_COLUMN`
41. `OP_SLICE_COLUMN_BACKWARD`
42. `OP_STACK_ROWS`
43. `OP_STACK_ROWS_BACKWARD`
44. `OP_CONCATENATE`
45. `OP_CONCATENATE_BACKWARD`
46. `OP_SGD_UPDATE`
47. `OP_ADAM_UPDATE`
48. `OP_CLIP_GRAD_VALUE`
49. `OP_SUM_REDUCE`
50. `OP_SUM_REDUCE_BACKWARD`
51. `OP_SUM_REDUCE_COLUMNS`
52. `OP_ZERO_GRAD`
53. `OP_PAD2D`
54. `OP_ADD_SCALAR`
55. `OP_MAX_POOL_1D_FORWARD`
56. `OP_MAX_POOL_1D_BACKWARD`
57. `OP_MAX_POOL_2D_FORWARD`
58. `OP_MAX_POOL_2D_BACKWARD`
59. `OP_BATCH_NORM_1D_FORWARD`
60. `OP_BATCH_NORM_1D_BACKWARD`
61. `OP_BATCH_NORM_2D_FORWARD`
62. `OP_BATCH_NORM_2D_BACKWARD`
63. `OP_EMBEDDING_FORWARD`
64. `OP_EMBEDDING_BACKWARD`
65. `OP_PAD2D_BACKWARD`
66. `OP_SCALE_MATRIX`
67. `OP_SCALE_MATRIX_BACKWARD`


## Methods

* **`read_tape<T>(const uint8_t* tape, int& offset)`**
  A templated utility function that safely extracts primitive data types (such as integers, floats, and booleans) from the current position in the byte array. It automatically advances the tape offset by the exact byte size of the extracted type.


* **`read_string(const uint8_t* tape, int& offset)`**
  A utility function that extracts a dynamically sized string from the byte array. It reads a length-prefixed format by first extracting a 2-byte unsigned integer for the length, then reading that exact number of characters to form the string, and finally advancing the offset.


* **`GraphExecutor::run_tape(const uint8_t* tape, int total_bytes, bool sync)`**
  The core execution loop. It pulls a stream from the CUDA manager and iterates through the tape until all bytes are processed. For each instruction, it reads a 4-byte OpCode, enters a switch statement to decode the specific parameters (names, indices, dimensions) required for that operation, and calls the appropriate method on the `CudaManager`. If the `sync` parameter is true, it explicitly synchronizes the CUDA stream before the method completes to ensure all GPU operations have finished.

  
# manager

The `manager` class is the central execution powerhouse of the execution engine. It manages Tensors, CudaStreams and is responsible to handle `cuBLAS` and custom
CUDA kernels. 

### memory management: 
# CUDA Manager

The `CudaManager` class serves as the primary engine for GPU resource management and kernel execution. It directly maintains the state of the GPU by managing a `memory` registry, which is a map of string names to `Gpu_Element` structures containing raw device pointers and tensor dimensions. To handle asynchronous execution, it utilizes a `stream_pool` of `cudaStream_t` objects and a `cublas_handle` for high-performance linear algebra operations. Access to these shared resources is synchronized via a `cuda_mutex` to ensure thread safety during allocation and execution.



## Available Methods

* **`CudaManager()` / `~CudaManager()`**: Initializes and destroys the cuBLAS handle and the pool of concurrent CUDA streams.
* **`get_stream_from_pool()`**: Fetches an available CUDA stream from the internal pool for asynchronous task dispatch.


* **`synchronize_stream(cudaStream_t stream)`**: Blocks the host thread until all operations currently queued in the specified CUDA stream are completed.
* **`allocate(const std::string& name, const std::vector<int>& sizes)`**: Allocates a block of device memory for a tensor of the specified dimensions and registers it in the internal memory map.
* **`exists(const std::string& name) const`**: Returns a boolean indicating whether a tensor with the given name is currently allocated on the GPU.
* **`free(const std::string& name)`**: Deallocates the GPU memory associated with a specific tensor name and removes it from the registry.
* **`copyHostToDevice(const std::string& name, const float* h_ptr)`**: Transfers data from a host-side float array to the allocated device memory of the specified tensor.
* **`retrieve(const std::string& name)`**: Pulls data from a GPU tensor back to the host, returning it as a vector of floats.
* **`get(const std::string& name)`**: Directly returns the raw float pointer pointing to the memory address of the tensor on the GPU.
* **`get_element(const std::string& name)`**: Retrieves the full `Gpu_Element` structure, including the pointer and the shape of the tensor.
* **`zero_grad(const std::string& name, cudaStream_t stream)`**: Sets all elements within a specified tensor to zero, typically used for clearing gradients.
* **`pad2d(const std::string& name_in, const std::string& name_out, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream)`**: Performs 2D spatial padding on an input tensor based on the specified top, bottom, left, and right margins.
* **`pad2d_backward(const std::string& name_grad_out, const std::string& name_grad_in, int pad_t, int pad_b, int pad_l, int pad_r, cudaStream_t stream)`**: Reverses the 2D padding operation during the backward pass to distribute gradients back to the original input dimensions.
* **`add_scalar(const std::string& name_in, const std::string& name_out, float scalar, cudaStream_t stream)`**: Adds a constant scalar value to every element in the input tensor.
* **`fill(const std::string& name, float value, cudaStream_t stream)`**: Assigns a specific float value to every element in the targeted device tensor.
* **`copy(const std::string& name_src, const std::string& name_dest, cudaStream_t stream)`**: Performs a device-to-device copy of data from a source tensor to a destination tensor.
* **`add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream)`**: Performs element-wise addition of two tensors and stores the result in a third.
* **`add_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream)`**: Performs in-place element-wise addition, adding the source tensor directly into the destination tensor.
* **`subtract(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream)`**: Performs element-wise subtraction of tensor B from tensor A.
* **`subtract_into(const std::string& name_src, const std::string& name_dest, cudaStream_t stream)`**: Performs in-place element-wise subtraction, subtracting the source tensor from the destination.
* **`multiply(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream)`**: Executes element-wise multiplication of two tensors.
* **`divide(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream)`**: Executes element-wise division of tensor A by tensor B.
* **`exp_elementwise(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Computes the natural exponential function for every element in the tensor.
* **`multiply_backward(const std::string& grad_out_name, const std::string& other_input_name, const std::string& grad_in_name, cudaStream_t stream)`**: Calculates the gradient for element-wise multiplication during backpropagation.
* **`divide_backward(const std::string& A_name, const std::string& B_name, const std::string& grad_out_name, const std::string& grad_A_name, const std::string& grad_B_name, cudaStream_t stream)`**: Calculates the gradients for both inputs involved in an element-wise division.
* **`exp_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradient for the exponential function pass.
* **`matmul(const std::string& name_A, const std::string& name_B, const std::string& name_C, bool transpose_A, bool transpose_B, float alpha, float beta, bool use_tensor_cores, cudaStream_t stream)`**: Executes matrix multiplication using cuBLAS, with optional transpositions, scaling factors, and Tensor Core acceleration.
* **`transpose(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Computes the transpose of a 2D matrix.
* **`broadcast_add(const std::string& name_A, const std::string& name_B, const std::string& name_C, cudaStream_t stream)`**: Adds a smaller tensor to a larger one using broadcasting rules, typically for adding a bias vector to a matrix.
* **`relu(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Applies the Rectified Linear Unit activation function element-wise.
* **`relu_backward(const std::string& name_in, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradient of the ReLU activation for backpropagation.
* **`sigmoid(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Applies the Sigmoid activation function element-wise.
* **`sigmoid_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradient of the Sigmoid activation.
* **`tanh(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Applies the Hyperbolic Tangent activation function element-wise.
* **`tanh_backward(const std::string& name_out_data, const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradient of the Tanh activation.
* **`softmax_forward(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Performs the forward softmax normalization on the input tensor.
* **`softmax_backward(const std::string& name_grad_out, const std::string& name_out_data, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradients for the softmax operation.
* **`mse_loss_forward(const std::string& pred_name, const std::string& target_name, const std::string& out_name, cudaStream_t stream)`**: Computes the Mean Squared Error loss between predictions and targets.
* **`mse_loss_backward(const std::string& pred_name, const std::string& target_name, const std::string& grad_in_name, const std::string& grad_out_name, cudaStream_t stream)`**: Computes the gradient of the MSE loss.
* **`bce_loss_forward(const std::string& name_pred, const std::string& name_target, const std::string& name_out, cudaStream_t stream)`**: Computes the Binary Cross-Entropy loss.
* **`bce_loss_backward(const std::string& name_grad_out, const std::string& name_pred, const std::string& name_target, const std::string& name_grad_in, cudaStream_t stream)`**: Computes the gradient of the BCE loss.
* **`sgd_update(const std::string& data_name, const std::string& grad_name, float learning_rate, cudaStream_t stream)`**: Updates tensor data in-place using Stochastic Gradient Descent based on the provided gradients and learning rate.
* **`adam_update(...)`**: Executes the Adam optimization step, updating parameters using first and second moment estimates, weight decay, and step counting.
* **`clip_grad_value(const std::string& buffer_name, float clip_value, cudaStream_t stream)`**: Clamps gradient values within a specified range to prevent exploding gradients.
* **`matmul_bias_relu_forward(...)`**: A fused operation that performs a matrix multiplication, adds a bias vector, and applies a ReLU activation in a single kernel pass.
* **`layer_norm_forward(...)`**: Computes forward Layer Normalization, calculating and storing means and reciprocal standard deviations.
* **`layer_norm_backward(...)`**: Computes the backward pass for Layer Normalization, providing gradients for input, gamma, and beta parameters.
* **`conv2d_forward(const std::string& name_in, const std::string& name_kernel, const std::string& name_out, cudaStream_t stream)`**: Executes a 2D convolution forward pass.
* **`conv2d_backward_input(...)`**: Computes the gradient of the loss with respect to the input of a 2D convolution.
* **`conv2d_backward_kernel(...)`**: Computes the gradient of the loss with respect to the kernels of a 2D convolution.
* **`im2col(...)`**: Unrolls image patches into a column matrix to facilitate convolution through matrix multiplication.
* **`col2im(...)`**: Aggregates column-wise gradients back into an image-shaped tensor.
* **`max_pool_1d_forward(...)`**: Performs 1D max pooling and stores the indices of the maximum values for the backward pass.
* **`max_pool_1d_backward(...)`**: Reconstructs the gradient for 1D max pooling using the stored indices.
* **`max_pool_2d_forward(...)`**: Performs 2D spatial max pooling and stores the argmax indices.
* **`max_pool_2d_backward(...)`**: Reconstructs the gradient for 2D spatial max pooling.
* **`batch_norm_1d_forward(...)`**: Computes 1D Batch Normalization and maintains running statistics.
* **`batch_norm_1d_backward(...)`**: Computes the backward pass for 1D Batch Normalization.
* **`batch_norm_2d_forward(...)`**: Computes 2D spatial Batch Normalization.
* **`batch_norm_2d_backward(...)`**: Computes the backward pass for 2D spatial Batch Normalization.
* **`scale_matrix(const std::string& name_in, const std::string& name_out, float scale, cudaStream_t stream)`**: Multiplies all elements of a matrix by a scalar scale factor.
* **`scale_matrix_backward(const std::string& name_grad_out, const std::string& name_grad_in, float scale, cudaStream_t stream)`**: Calculates the gradient for the matrix scaling operation.
* **`slice_row(const std::string& name_in, const std::string& name_out, int row_index, cudaStream_t stream)`**: Extracts a single row from a 2D matrix into a new tensor.
* **`slice_row_backward(const std::string& name_grad_out, const std::string& name_grad_in, int row_index, cudaStream_t stream)`**: Places gradients from a sliced row back into the full-sized gradient tensor.
* **`slice_column(const std::string& name_in, const std::string& name_out, int start_col, int end_col, cudaStream_t stream)`**: Extracts a specific range of columns from a matrix.
* **`slice_column_backward(const std::string& name_grad_out, const std::string& name_grad_in, int start_col, int end_col, cudaStream_t stream)`**: Maps column-wise gradients back to the original input tensor shape.
* **`stack_rows(const std::vector<std::string>& names_in, const std::string& name_out, int axis, cudaStream_t stream)`**: Combines multiple input tensors into a single stacked tensor along the specified axis.
* **`stack_rows_backward(const std::string& name_grad_out, const std::vector<std::string>& names_grad_in, int axis, cudaStream_t stream)`**: Splits the gradient of a stacked tensor back into individual gradients for each input.
* **`concatenate(const std::string& name_A, const std::string& name_B, const std::string& name_out, int axis, cudaStream_t stream)`**: Joins two tensors end-to-end along the specified dimension.
* **`concatenate_backward(const std::string& name_grad_out, const std::string& name_grad_in_A, const std::string& name_grad_in_B, int axis, int split_index, cudaStream_t stream)`**: Splits the gradient of a concatenated tensor back into gradients for the original components.
* **`sum_reduce(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Reduces the entire input tensor to a single scalar sum.
* **`sum_reduce_backward(const std::string& name_grad_out, const std::string& name_grad_in, cudaStream_t stream)`**: Broadcasts a scalar gradient back to the shape of the original input.
* **`sum_reduce_columns(const std::string& name_in, const std::string& name_out, cudaStream_t stream)`**: Sums elements along the columns of a matrix, resulting in a column vector.
* **`embedding_forward(const std::string& name_indices, const std::string& name_weight, const std::string& name_out, cudaStream_t stream)`**: Retrieves dense vectors from a weight matrix based on integer indices.
* **`embedding_backward(const std::string& name_grad_out, const std::string& name_indices, const std::string& name_grad_weight, cudaStream_t stream)`**: Accumulates gradients from the output into the embedding weights at the specified indices.