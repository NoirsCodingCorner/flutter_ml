### Data Loading & Storage

* **`OP_LOAD_SAMPLE`**
* `name_in` (String)
* `name_out` (String)
* `sample_idx` (Int32)


* **`OP_STORE_SAMPLE`**
* `name_in` (String)
* `name_dest` (String)
* `sample_idx` (Int32)


* **`OP_COPY`**
* `name_in` (String)
* `name_out` (String)


* **`OP_FILL`**
* `name_out` (String)
* `value` (Float)



### Basic Arithmetic

* **`OP_ADD`**
* `name_A` (String), `name_B` (String), `name_C` (String)


* **`OP_ADD_INTO`**
* `name_src` (String), `name_dest` (String)


* **`OP_SUBTRACT`**
* `name_A` (String), `name_B` (String), `name_C` (String)


* **`OP_SUBTRACT_INTO`**
* `name_src` (String), `name_dest` (String)


* **`OP_MULTIPLY`**
* `name_A` (String), `name_B` (String), `name_C` (String)


* **`OP_MULTIPLY_BACKWARD`**
* `name_grad_out` (String), `name_other_in` (String), `name_grad_in` (String)


* **`OP_DIVIDE`**
* `name_A` (String), `name_B` (String), `name_C` (String)


* **`OP_DIVIDE_BACKWARD`**
* `name_A` (String), `name_B` (String), `name_grad_out` (String), `name_grad_A` (String), `name_grad_B` (String)



### Matrix Operations

* **`OP_MATMUL`**
* `name_A` (String), `name_B` (String), `name_C` (String)
* `trans_A` (Bool), `trans_B` (Bool)
* `alpha` (Float), `beta` (Float)
* `t_cores` (Bool)


* **`OP_TRANSPOSE`**
* `name_in` (String), `name_out` (String)


* **`OP_BROADCAST_ADD`**
* `name_A` (String), `name_B` (String), `name_C` (String)



### Activations

* **`OP_RELU`**
* `name_in` (String), `name_out` (String)


* **`OP_RELU_BACKWARD`**
* `name_in` (String), `name_grad_out` (String), `name_grad_in` (String)


* **`OP_SIGMOID`**
* `name_in` (String), `name_out` (String)


* **`OP_SIGMOID_BACKWARD`**
* `name_out_data` (String), `name_grad_out` (String), `name_grad_in` (String)


* **`OP_TANH`**
* `name_in` (String), `name_out` (String)


* **`OP_TANH_BACKWARD`**
* `name_out_data` (String), `name_grad_out` (String), `name_grad_in` (String)


* **`OP_EXP_ELEMENTWISE`**
* `name_in` (String), `name_out` (String)


* **`OP_EXP_BACKWARD`**
* `name_grad_out` (String), `name_out_data` (String), `name_grad_in` (String)


* **`OP_SOFTMAX_FORWARD`**
* `name_in` (String), `name_out` (String)


* **`OP_SOFTMAX_BACKWARD`**
* `name_grad_out` (String), `name_out_data` (String), `name_grad_in` (String)



### Loss Functions

* **`OP_MSE_LOSS_FORWARD`**
* `name_pred` (String), `name_target` (String), `name_out` (String)


* **`OP_MSE_LOSS_BACKWARD`**
* `name_pred` (String), `name_target` (String), `name_grad_in` (String), `name_grad_out` (String)


* **`OP_BCE_LOSS_FORWARD`**
* `name_pred` (String), `name_target` (String), `name_out` (String)


* **`OP_BCE_LOSS_BACKWARD`**
* `name_grad_out` (String), `name_pred` (String), `name_target` (String), `name_grad_in` (String)



### Advanced Layers (Conv / Fused / Norm)

* **`OP_MATMUL_BIAS_RELU_FORWARD`**
* `name_X` (String), `name_W` (String), `name_B` (String), `name_Relu_Out` (String), `name_PreR_Out` (String)


* **`OP_LAYER_NORM_FORWARD`**
* `name_in` (String), `name_gamma` (String), `name_beta` (String), `name_out` (String), `name_mean` (String), `name_rstd` (String)
* `eps` (Float)


* **`OP_LAYER_NORM_BACKWARD`**
* `name_grad_out` (String), `name_in` (String), `name_gamma` (String), `name_mean` (String), `name_rstd` (String), `name_grad_in` (String), `name_grad_gamma` (String), `name_grad_beta` (String)


* **`OP_CONV2D_FORWARD`**
* `name_in` (String), `name_kernel` (String), `name_out` (String)


* **`OP_CONV2D_BACKWARD_INPUT`**
* `name_grad_out` (String), `name_kernel` (String), `name_grad_in` (String)


* **`OP_CONV2D_BACKWARD_KERNEL`**
* `name_in` (String), `name_grad_out` (String), `name_grad_kernel` (String)


* **`OP_IM2COL`**
* `name_in` (String), `name_out` (String)
* `kh` (Int32), `kw` (Int32)


* **`OP_COL2IM`**
* `name_col_grad` (String), `name_in_grad` (String)
* `kh` (Int32), `kw` (Int32)



### Tensor Manipulation

* **`OP_SLICE_ROW`**
* `name_in` (String), `name_out` (String)
* `row` (Int32)


* **`OP_SLICE_ROW_BACKWARD`**
* `name_grad_out` (String), `name_grad_in` (String)
* `row` (Int32)


* **`OP_SLICE_COLUMN`**
* `name_in` (String), `name_out` (String)
* `start` (Int32), `end` (Int32)


* **`OP_SLICE_COLUMN_BACKWARD`**
* `name_grad_out` (String), `name_grad_in` (String)
* `start` (Int32), `end` (Int32)


* **`OP_STACK_ROWS`**
* `count` (Int32)
* `names_in` (Array of Strings, length = count)
* `name_out` (String)
* `axis` (Int32)


* **`OP_STACK_ROWS_BACKWARD`**
* `name_grad_out` (String)
* `count` (Int32)
* `names_grad_in` (Array of Strings, length = count)
* `axis` (Int32)


* **`OP_CONCATENATE`**
* `name_A` (String), `name_B` (String), `name_out` (String)
* `axis` (Int32)


* **`OP_CONCATENATE_BACKWARD`**
* `name_grad_out` (String), `name_grad_inA` (String), `name_grad_inB` (String)
* `axis` (Int32), `split` (Int32)



### Optimizers

* **`OP_SGD_UPDATE`**
* `name_data` (String), `name_grad` (String)
* `lr` (Float)


* **`OP_ADAM_UPDATE`**
* `name_data` (String), `name_grad` (String), `name_m` (String), `name_v` (String)
* `lr` (Float), `b1` (Float), `b2` (Float), `eps` (Float)
* `step` (Int32)
* `wd` (Float)


* **`OP_CLIP_GRAD_VALUE`**
* `name_buffer` (String)
* `clip_val` (Float)



### Reductions

* **`OP_SUM_REDUCE`**
* `name_in` (String), `name_out` (String)


* **`OP_SUM_REDUCE_BACKWARD`**
* `name_grad_out` (String), `name_grad_in` (String)


* **`OP_SUM_REDUCE_COLUMNS`**
* `name_in` (String), `name_out` (String)