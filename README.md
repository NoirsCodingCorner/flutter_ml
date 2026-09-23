# flutter\_ml

[](https://www.google.com/search?q=https://pub.dev/packages/flutter_ml)
[](https://opensource.org/licenses/MIT)

A deep learning library for Dart, built from scratch, finally bringing machine learning to the language without python ffi calls!


---
UPDATE NOTE:
Version 2.0 features advanced GPU computation, allowing for a significant speed up of standard operations.
The safetensor and standard BERT- Modules are supported out of the Box as well. 
For implementing custom architectures, using the individual math operations and layer functionalities is heavily encouraged.
---

Since 2.0 this package is divided into two main components, consisting of a pure dart autograd and a GPU-accelerated implementation.
The accelerated version allows for the developing of accelerated Machine Learning algorithms for CUDA v.12+ compatible devices.
Future integration of WEBGPU to allow seamless integration of mobile and web based applications is planned.

## General Features
This package includes a wide range of features, from the low-level engine to high-level model APIs.
The following features are universally supported:
  - **Autograd:** An auto-gradient engine built from scratch designed specifically for deep learning.
  - **Dynamic computation graphs** A special feature of this design is totally transparent printouts of the resulting math
structure, to allow for precise inspection of the autograd engine's execution.
  - **Wide Range of Model types** Most standard model architectures like `Transformer` , `Dense` , `Conv` , `LSTM` are supported
  - **Direct Tensor Manipulation** Tensors can be directly accessed for full control. There is nothing stopping the development. ( But also nothing preventing mistakes )


## CPU - Dart components
The version `1.0.0` operations of the autograd are still fully supported, allowing for a multi platform development of 
advanced on device Machine Learning features.

The following Layers (similar to torch modules) are supported:
* `DenseLayer`: Standard fully-connected layer for `Vector` data.
* `DenseLayerMatrix`: Batch-processed fully-connected layer for `Matrix` data.
* `RNN`: A simple Recurrent Neural Network (RNN) layer.
* `LSTMLayer`: A Long Short-Term Memory (LSTM) recurrent layer.
* `DualLSTMLayer`: A custom hierarchical LSTM with two internal tiers (fast and slow).
* `MultiTierLSTMLayer`: A generalized, hierarchical LSTM with an arbitrary number of configured timescales.
* `GeneralizedChainedScaleLayer`: A self-contained, multi-scale recurrent layer for processing high-frequency data.
* `Conv2DLayer`: A 2D convolutional layer for 2D `Matrix` inputs.
* `ConvLSTMLayer`: A Convolutional LSTM layer for spatiotemporal data (like video).
* `MaxPooling1DLayer` / `MaxPooling2DLayer`
* `AveragePooling2DLayer`
* `GlobalAveragePoolingLayer` / `GlobalAveragePooling1D`
* `EmbeddingLayer` / `EmbeddingLayerMatrix`: Converts token indices to dense vectors.
* `PositionalEncoding`: Injects sinusoidal position information.
* `SingleHeadAttention`: A single head of the self-attention mechanism.
* `MultiHeadAttention`: Runs multiple `SingleHeadAttention` heads in parallel.
* `TransformerEncoderBlock`: A full encoder block (Attention + FeedForward).
* `BatchNorm1D` / `BatchNorm2D`: Batch Normalization for 1D and 3D data.
* `LayerNormalization` / `LayerNormalizationVector`: Layer Normalization for `Matrix` and `Vector` data.
* `DropoutLayer` / `DropoutLayerMatrix`: Regularization layer for `Vector` and `Matrix` inputs.
* `FlattenLayer`: Reshapes a `Matrix` into a `Vector`.
* `ReshapeVectorToMatrixLayer`: Reshapes a `Vector` to a 1x1 `Matrix`.
* `ReLULayer` / `ReLULayerMatrix`

#### Available Optimizer Methods:
* `SGD`: The standard Stochastic Gradient Descent.
* `Momentum`: `SGD` with the addition of a momentum (velocity) term.
* `NAG`: Nesterov Accelerated Gradient, an improvement on `Momentum`.
* `Adagrad`: An adaptive optimizer good for sparse data.
* `RMSprop`: An adaptive optimizer that performs well with RNNs.
* `Adam`: The most common, general-purpose adaptive optimizer.
* `AMSGrad`: A variant of `Adam` that fixes a potential convergence issue.
* `AdamW`: A variant of `Adam` that improves weight decay (L2 regularization).

#### Available Activation Functions:
* `ReLU` : Standard ReLU functions
* `LeakyReLU` : A leaky version of Relu dropping outputs to prevent Overfitting
* `ELU` : Standard activation function
* `Sigmoid` : (For binary classification outputs)
* `Softmax` : (For multi-class classification outputs)
* `SiLU` : SiLU activation function
* `Swish` : Swish activation function
* `Mish` : Mish activation function

#### Available Math Functions for Tensor calculations:

Scalar:
* `padMatrix` : Add empty borders to a matrix
* `add` : Adds two numbers together
* `multiply` : Multiplies two Scalars together
* `sigmoidScalar` : Applies the sigmoid function to a Scalar
* `binaryCrossEntropy` : Applies BCE function to a Scalar

Vector
* `addVector` : Adds two vectors element wise together
* `addScalar` : Adds a Scalar to all elements in a Vector
* `concatenate` : Chains two Vector Tensors together into a big Vector
* `dot` : Calculates the Scalar dot product of two Vector Tensors
* `elementWiseMultiply` : Multiplies two same sized Vectors together via their indices
* `mse` : Calculate the Scalar value of the Mean Squared Error between two vectors
* `relu` : Applies a ReLU function to a Vector
* `sigmoid` : Applies a sigmoid function to a Vector
* `sum` : Sums up all elements in a Vector into a single Scalar
* `vectorTanh` : Applies a Tanh function to each element in a Vector
* `vectorExp` : Applies the natural  exponential function to each element in a Vector
* `vectorLog` : Applies the natural logarithm function to each element in a Vector
* `softplus` : Applies the softplus function to a Vector
* `batchNorm1dMath` : Normalizes a given Vector with gamma, beta, running mean, running variance, feature number, momentum and epsilon
* `dropoutVectorMath` : Applies a dropout rate (setting values to 0) inside the given Vector
* `maxPool1d` : Gets the Max inside a given poolSize of a Vector
* `softmaxVector` : Applies a softmax function over a given Vector
* `swishVector` : Applies a swish function over a given Vector
* `eluVector` : Applies an elu function over a given Vector
* `leakyReluVector` : Applies a leakyRelu function on every element of a Vector
* `mishVector` : Applies a mish function on every element of a Vector

Matrix
* `addMatrix` : Adds two Matrices of same size together element wise
* `addMatrixAndVector` : Adds a Vector to every Row of a Matrix
* `addScalarToMatrix` : Adds a Scalar to every element of a Matrix
* `concatenateMatricesByColumn` : Chains two Matrices together in Column order
* `elementWiseMultiplyMatrix` : Multiplies two same sized Matrices together
* `conv2d` : Applies a standard 2D convolution kernel sliding over an input Matrix
* `matMul` : The heart of most modern ML, multiplies each element of a matrix with each element of another Matrix
* `mseMatrix` : Applies the MSE function over two Matrices to find the MSE between them
* `reluMatrix` : Applies the ReLU function to every element of a Matrix
* `reshapeVectorToMatrix` : Takes a Vector and slices it into the specified number of rows and columns
* `scaleMatrix` : Direct applies a decimal multiplication on every element of a Matrix
* `selectRow` : Selects a row from a given Matrix
* `sigmoidMatrix` : Applies the sigmoid function to every element in a Matrix
* `sumMatrix` : Sums up all elements in a Matrix into a single Scalar
* `tanhMatrix` : Applies the tanh function to every element of a Matrix
* `transpose` : Transposes (Swaps row and columns) of a Matrix
* `softmaxMatrix` : Applies the softmax function to a Matrix (tuning it into probabilities)
* `avgPool2d` : Applies a 2D average Pool function onto a Matrix with a given stride and poolSize
* `dropoutMatrixMath` : Applies a dropout rate (setting values to 0) inside the given Matrix
* `maxPool2d` : Gets the Max inside a given poolSize of a Vector
* `eluMatrix` : Applies and elu function over every element of a Matrix
* `leakyReluMatrix` : Applies a leakyRelu function on every element of a Matrix
* `mishMatrix` : Applies a mish function on every element of a Matrix

* Tensor3D (A list of Matrices)
* `add3D` : Adds two Tensor3D Tensors element wise together
* `elementWiseMultiply3D` : Multiplies two Tensor3D Tensors element wise
* `concatenate3D` : Chains Two Tensor3D Tensors together over their highest dimension
* `batchNorm2dMath` : Normalizes a given Tensor3D with gamma, beta, running mean, running variance, feature number,
  momentum and epsilon
* `stackMatricesTo3D` : Allows stacking Matrices together to a Tensor3D

#### How to use:
```dart
import 'package:flutter_ml/full_library.dart';

void main(){
  // Initialise the Tensors with the wanted sizes and values
  Tensor<Vector>VecA=Tensor([1.1, 2.2, 3.3]);
  Tensor<Vector>VecB=Tensor([4.4, 5.5, 6.6]);
  Tensor<Vector>VecC=Tensor([1.0,0.1,-1.0]);
  
  //  Use the available Math functions to calculate
  Tensor<Vector>VecD=addVector(VecA, VecB);
  Tensor<Vector>VecE=elementWiseMultiply(VecC, VecD);
  
  //  Retrieval of the result
  print(VecE.value);
  
  //  Print the computational Graph that was used to generate the resulting Vector
  VecE.printGraph();
}
```

This results in:
```
📊 Computational Graph [Hybrid CPU/GPU]:
└──  t_4 [3] [CPU] (Op: multiply_vector)
    ├──  t_2 [3] [CPU] (Leaf: Input)
    └──  t_3 [3] [CPU] (Op: add_vector)
        ├──  t_0 [3] [CPU] (Leaf: Input)
        └──  t_1 [3] [CPU] (Leaf: Input)
        
[5.5, 0.7699999809265137, -9.899999618530273]
```

#### Example Neural Network:

```dart
import 'package:flutter_ml/full_library.dart';

Future<void> main() async {
  //  Initialize Inputs
  List<Vector> xorInputs = [];
  xorInputs.add([0.0, 0.0]);
  xorInputs.add([0.0, 1.0]);
  xorInputs.add([1.0, 0.0]);
  xorInputs.add([1.0, 1.0]);

  //  Initialize Targets
  List<Vector> xorTargets = [];
  xorTargets.add([0.0]);
  xorTargets.add([1.0]);
  xorTargets.add([1.0]);
  xorTargets.add([0.0]);
  
  //  Create and load Layers
  List<Layer<dynamic, dynamic>> layers = [];
  layers.add(DenseLayer(8, activation: ReLU()));
  layers.add(DenseLayer(1, activation: Sigmoid()));
  
  //  Create and load the model via a simple SNetwork (Sequential Network)
  SNetwork model = SNetwork(layers, name: 'XOR-Net');
  Tensor<Vector> initialInputTensor = Tensor<Vector>(xorInputs[0]);
  model.predict(initialInputTensor);
  
  //  Choose and create the Optimizer
  SGD optimizer = SGD(model.parameters, learningRate: 0.1);
  model.compile(configuredOptimizer: optimizer);

  int epochs = 5000;
  print('Training ${model.name} for $epochs epochs...');
  
  // Initial run to build the model before SNET does hard training
  model.fit(xorInputs, xorTargets, epochs: epochs, debug: true);

  // Save and load example for how models can be saved and loaded
  String modelPath = 'xor_model.json';
  await model.save(modelPath);

  print('\n--- Loading weights into new model ---');

  List<Layer<dynamic, dynamic>> loadedLayers = [];
  // Ensure the loaded architecture matches exactly
  loadedLayers.add(DenseLayer(8, activation: ReLU()));
  loadedLayers.add(DenseLayer(1, activation: Sigmoid()));

  SNetwork loadedModel = SNetwork(loadedLayers, name: 'Loaded-XOR-Net');
  loadedModel.predict(initialInputTensor);
  await loadedModel.load(modelPath);

  print('\n--- Testing Predictions (from LOADED model) ---');
  
  //  Starts running the actual training loop
  int i = 0;
  for (int j = 0; j < xorInputs.length; j = j + 1) {
    Vector input = xorInputs[j];
    Tensor<Vector> inputTensor = Tensor<Vector>(input);
    Tensor<Vector> predictionTensor = loadedModel.predict(inputTensor) as Tensor<Vector>;

    int target = xorTargets[i][0].toInt();
    double rawOutput = predictionTensor.value[0];
    int predictedClass = (rawOutput > 0.5) ? 1 : 0;
    bool isCorrect = (predictedClass == target);

    print('Input: $input, Target: $target, Output: ${rawOutput.toStringAsFixed(4)}, Predicted: $predictedClass, Correct: $isCorrect');
    i = i + 1;
  }
}
```
Result:
```
Shell: Epoch 4998/5000: [====================>] 100%, Avg Loss: 0.000160
Shell: Epoch 4999/5000: [=====>               ] 25%
Shell: Epoch 4999/5000: [==========>          ] 50%
Shell: Epoch 4999/5000: [===============>     ] 75%
Shell: Epoch 4999/5000: [====================>] 100%
Shell: Epoch 4999/5000: [====================>] 100%, Avg Loss: 0.000160
```
With the structure: 
```
📊 Computational Graph [Hybrid CPU/GPU]:
└──  t_180057 [] [CPU] (Op: mse_vector)
    ├──  t_180056 [1] [CPU] (Op: sigmoid_vector)
    │   └──  t_180055 [1] [CPU] (Op: add_vector)
    │       ├──  t_180054 [1] [CPU] (Op: matVecMul)
    │       │   ├──  t_6 [1, 8] [CPU] (Leaf: Input)
    │       │   └──  t_180053 [8] [CPU] (Op: relu_vector)
    │       │       └──  t_180052 [8] [CPU] (Op: add_vector)
    │       │           ├──  t_180051 [8] [CPU] (Op: matVecMul)
    │       │           │   ├──  t_1 [8, 2] [CPU] (Leaf: Input)
    │       │           │   └──  t_180049 [2] [CPU] (Leaf: Input)
    │       │           └──  t_2 [8] [CPU] (Leaf: Input)
    │       └──  t_7 [1] [CPU] (Leaf: Input)
    └──  t_180050 [1] [CPU] (Leaf: Input)
```

## GPU - Accelerated Components

Since version `2.0.0` an additional system for advanced GPU Programming directly inside of dart has been established.
It the current form it features Cuda 12.1+ support.

In contrast to direct eager execution of CPU components and functions, the GPU functionality allows for building and execution
of `CommandBuffer` - tapes, usually referred to as `OPTapes`(Operation Tapes).
These tapes allow for buffered execution to build models once and run the same operations whenever needed without graph-building delays,
which is a feature used in many modern Autograd Frameworks. Every feature is encoded in an Int32 Code, meaning a total possible amount of
2,147,483,647 different operations.
Currently available OPCodes: 

```dart
// --- 0 - 99: Data & Memory Management ---
const int OP_LOAD_SAMPLE              = 2;
const int OP_STORE_SAMPLE             = 4;
const int OP_COPY                     = 6;
const int OP_FILL                     = 8;
const int OP_ZERO_GRAD                = 10;

// --- 100 - 199: Basic Math (Scalars & Element-wise) ---
const int OP_ADD                      = 100;
const int OP_ADD_INTO                 = 102;
const int OP_ADD_SCALAR               = 104;
const int OP_SUBTRACT                 = 106;
const int OP_SUBTRACT_INTO            = 108;
const int OP_MULTIPLY                 = 110;
const int OP_MULTIPLY_BACKWARD        = 111;
const int OP_DIVIDE                   = 112;
const int OP_DIVIDE_BACKWARD          = 113;
const int OP_EXP_ELEMENTWISE          = 114;
const int OP_EXP_BACKWARD             = 115;
const int OP_LOG_ELEMENTWISE          = 116;
const int OP_LOG_BACKWARD             = 117;
const int OP_ABS_ELEMENTWISE          = 118;
const int OP_ABS_BACKWARD             = 119;
const int OP_SQRT_ELEMENTWISE         = 120;
const int OP_SQRT_BACKWARD            = 121;
const int OP_POW_ELEMENTWISE          = 122;
const int OP_POW_BACKWARD             = 123;
const int OP_CLAMP_ELEMENTWISE        = 124;
const int OP_CLAMP_BACKWARD           = 125;

// --- 200 - 299: Matrix Operations ---
const int OP_MATMUL                   = 200;
const int OP_TRANSPOSE                = 202;
const int OP_BROADCAST_ADD            = 204;
const int OP_SCALE_MATRIX             = 206;
const int OP_SCALE_MATRIX_BACKWARD    = 207;

// --- 300 - 399: Activations ---
const int OP_RELU                     = 300;
const int OP_RELU_BACKWARD            = 301;
const int OP_SIGMOID                  = 302;
const int OP_SIGMOID_BACKWARD         = 303;
const int OP_TANH                     = 304;
const int OP_TANH_BACKWARD            = 305;
const int OP_GELU_FORWARD             = 306;
const int OP_GELU_BACKWARD            = 307;
const int OP_SOFTMAX_FORWARD          = 308;
const int OP_SOFTMAX_BACKWARD         = 309;

// --- 400 - 499: Loss Functions ---
const int OP_MSE_LOSS_FORWARD         = 400;
const int OP_MSE_LOSS_BACKWARD        = 401;
const int OP_BCE_LOSS_FORWARD         = 402;
const int OP_BCE_LOSS_BACKWARD        = 403;

// --- 500 - 599: Optimizers ---
const int OP_SGD_UPDATE               = 500;
const int OP_ADAM_UPDATE              = 502;
const int OP_CLIP_GRAD_VALUE          = 504;

// --- 600 - 699: Reductions ---
const int OP_SUM_REDUCE               = 600;
const int OP_SUM_REDUCE_BACKWARD      = 601;
const int OP_SUM_REDUCE_COLUMNS       = 602;
const int OP_SUM_REDUCE_ROWS          = 604;
const int OP_EMBEDDING_FORWARD        = 606;
const int OP_EMBEDDING_BACKWARD       = 607;

// --- 700 - 799: Tensor Manipulation ---
const int OP_SLICE_ROW                = 700;
const int OP_SLICE_ROW_BACKWARD       = 701;
const int OP_SLICE_COLUMN             = 702;
const int OP_SLICE_COLUMN_BACKWARD    = 703;
const int OP_STACK_ROWS               = 704;
const int OP_STACK_ROWS_BACKWARD      = 705;
const int OP_CONCATENATE              = 706;
const int OP_CONCATENATE_BACKWARD     = 707;
const int OP_PAD2D                    = 708;
const int OP_PAD2D_BACKWARD           = 709;

// --- 800 - 999: Advanced Spatial & Sequence Layers ---
const int OP_CONV2D_FORWARD               = 800;
const int OP_CONV2D_BACKWARD_INPUT        = 801;
const int OP_CONV2D_BACKWARD_KERNEL       = 803;
const int OP_CONV2D_MULTI_FORWARD         = 804;
const int OP_CONV2D_MULTI_BACKWARD_INPUT  = 805;
const int OP_CONV2D_MULTI_BACKWARD_WEIGHT = 807;
const int OP_IM2COL                       = 808;
const int OP_COL2IM                       = 809;
const int OP_MAX_POOL_1D_FORWARD          = 810;
const int OP_MAX_POOL_1D_BACKWARD         = 811;
const int OP_MAX_POOL_2D_FORWARD          = 812;
const int OP_MAX_POOL_2D_BACKWARD         = 813;
const int OP_AVG_POOL_2D_FORWARD          = 814;
const int OP_AVG_POOL_2D_BACKWARD         = 815;
const int OP_GLOBAL_AVG_POOL_FORWARD      = 816;
const int OP_GLOBAL_AVG_POOL_BACKWARD     = 817;
const int OP_BATCH_NORM_1D_FORWARD        = 820;
const int OP_BATCH_NORM_1D_BACKWARD       = 821;
const int OP_BATCH_NORM_2D_FORWARD        = 822;
const int OP_BATCH_NORM_2D_BACKWARD       = 823;
const int OP_LAYER_NORM_FORWARD           = 824;
const int OP_LAYER_NORM_BACKWARD          = 825;
const int OP_DROPOUT_FORWARD              = 826;
const int OP_DROPOUT_BACKWARD             = 827;

const int OP_MARKOV_COUNT             = 900;
const int OP_MARKOV_NORMALIZE         = 902;
const int OP_MARKOV_PREDICT           = 904;

// --- 1000+: Fused Kernels ---
const int OP_MATMUL_BIAS_RELU_FORWARD = 1200;
```

Since using those operations individually is cumbersome the most common structures are bundled into 
`TapeLayer` modules. The currently available modules are:

* `AveragePooling2DGPU` : A Layer for average pooling
* `BatchNorm1DGPU` : A layer for 1D Batch Normalization
* `BatchNorm2DGPU` : A layer for 2D Batch Normalization
* `Conv2DTapeLayer` : A 2D Convolution Layer usually used for spatial models
* `ConvLSTMTapeLayer` : A 2D Convolutional layer to allow for spatial and time dependent analysis
* `DenseLayer` : A fully connected Linear Layer seen as the base building block for most models
* `DenseReluLayer` (A fused layer for better performance)
* `DropoutMatrixTapeLayer` : A layer to randomly drop values from a matrix, preventing overfitting
* `DropoutTapeLayer` : A layer to randomly drop values from a matrix usually used to prevent overfitting
* `DualLSTMTapeLayer` : A experimental layer for a two timescale LSTM for advanced time series analysis
* `EmbeddingMatrixTapeLayer` : embedding layer which is the basis for translating symbols to matrices
* `EmbeddingTapeLayer` : An embedding layer to translate from symbols to vector
* `FlattenLayer` : Transforms a given multidimensional array into a Vector
* `GeluLayerMatrixTapeLayer` : Applies a Gelu function over its input matrix
* `GeluLayerTapeLayer` : Applies a gelu function over its input vector
* `GlobalAveragePooling1DTapeLayer` : Applies a 1D global average Pooling over the inputs. Often used in Large Language Models
* `GlobalAveragePoolingGPU` : Applies a global average pooling over the inputs
* `LSTMTapeLayer` : Standard LSTM layer. Recurrently uses gates to control long term memory over time series
* `LayerNormalizationTapeLayer` : Normalises a layer which can be seen as conversion to probabilities.
* `MaxPooling1DTapeLayer` : Max pools its input Vector
* `MaxPooling2DTapeLayer`: Max pools its output Matrix
* `MultiHeadAttentionTapeLayer` : A layer to apply a fully functioning and accelerated multiheadattention over a matrix
* `PositionalEncodingTapeLayer` : A layer to encode position via frequency into its input
* `RNNTapeLayer` : A standard recurrent neural network for series analysis 
* `ReLULayerMatrixTapeLayer` : Applies a Relu activation function onto the input matrix
* `ReLULayerTapeLayer` : Applies a Relu activation function onto the input matrix
* `SingleHeadAttentionTapeLayer` : A single head attention mechanism
* `TransformerEncoderBlockTapeLayer` : A full combination of multihead, positional encoding etc to provide a ready to use layer for the transformer architecture

Those Layers can be used individually to allow easier tape build or with the use a of SNetworkGPU.
The SNetworkGPU works as illustrated in the following example for a simple Fully Connected Neural Network:

```dart
void main() async{
  //  Initialise Cuda engine
  await CudaEngine.initialize(debug: false);

  //  Example dataset for a simple XOr
  List<double> rawX = <double>[
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  ];
  List<double> rawY = <double>[
    0.0,
    1.0,
    1.0,
    0.0
  ];

  //  Declare the network
  SNetworkGPU net = SNetworkGPU();

  //  Add the wanted TapeLayers
  net.add(DenseReluTL(8));
  net.add(DenseTL(1));

  //  Tell the network what input and outputs can be expected between the Layers and the learning-rate
  net.compile(<int>[4, 2], <int>[4, 1], 0.1);
  
  //  Itterate over the epochs and train a single step on the given data and print the loss
  for (int epoch = 1; epoch <= 10000; epoch = epoch + 1) {
    double currentLoss = net.trainStep(rawX, rawY);

    if (epoch % 100 == 0) {
      print('Epoch $epoch | Loss: ${currentLoss.toStringAsFixed(6)}');
    }
  }

  print('\n--- Final Predictions ---');
  List<dynamic> predictions = net.predict(rawX);

  print('[0, 0] -> Target: 0.0 | Output: ${predictions[0][0].toStringAsFixed(4)}');
  print('[0, 1] -> Target: 1.0 | Output: ${predictions[1][0].toStringAsFixed(4)}');
  print('[1, 0] -> Target: 1.0 | Output: ${predictions[2][0].toStringAsFixed(4)}');
  print('[1, 1] -> Target: 0.0 | Output: ${predictions[3][0].toStringAsFixed(4)}');

  //  Free the memory allocate for this network
  net.free();
  CudaEngine.dispose();
}
```

For atomic use of mathematical operations, this package provides a wide range of math functions for the gpu,
most of which are similar to the corresponding cpu functions.

### Overview over all GPUTensor mathematical functions in `v2.0.0`
#### Available GPU Mathematical Functions:

**Data Management:**
* `reshapeVectorToMatrixGPU` : Reshapes a Vector into a Matrix with specified rows and columns
* `reshapeMatrixTo3DGPU` : Reshapes a Matrix into a 3D Tensor with specified channel, height, and width dimensions
* `reshape3DToMatrixGPU` : Reshapes a 3D Tensor into a Matrix with specified dimensions
* `flatten3DToMatrixGPU` : Flattens a 3D Tensor into a Matrix
* `loadSampleGPU` : Loads a single sample from a dataset Matrix by index

**Basic Math:**
* `addGPU` : Adds two Tensors element-wise together
* `addVectorGPU` : Adds two Vectors element-wise together
* `addMatrixGPU` : Adds two Matrices element-wise together
* `add3DGPU` : Adds two 3D Tensors element-wise together
* `subtractGPU` : Subtracts two Tensors element-wise
* `subtractVectorGPU` : Subtracts two Vectors element-wise
* `subtractMatrixGPU` : Subtracts two Matrices element-wise
* `subtract3DGPU` : Subtracts two 3D Tensors element-wise
* `multiplyGPU` : Multiplies two Tensors element-wise
* `multiplyScalarGPU` : Multiplies two Scalars together
* `elementWiseMultiplyGPU` : Multiplies two Vectors element-wise
* `elementWiseMultiply3DGPU` : Multiplies two 3D Tensors element-wise
* `elementWiseMultiplyMatrixGPU` : Multiplies two Matrices element-wise
* `divideGPU` : Divides two Tensors element-wise
* `divideVectorGPU` : Divides two Vectors element-wise
* `divideMatrixGPU` : Divides two Matrices element-wise
* `divide3DGPU` : Divides two 3D Tensors element-wise
* `vectorExpGPU` : Applies the exponential function to each element in a Vector
* `absGPU` : Applies the absolute value function to a Tensor
* `sqrtGPU` : Applies the square root function to a Tensor
* `logGPU` : Applies the natural logarithm function to a Tensor
* `powGPU` : Raises each element in a Tensor to a specified power
* `clampGPU` : Clamps all values in a Tensor between a minimum and maximum value

**Matrix Operations:**
* `matMulGPU` : Multiplies two Matrices together via matrix multiplication
* `matVecMulGPU` : Multiplies a Matrix with a Vector
* `transposeGPU` : Transposes a Matrix (swaps rows and columns)
* `addMatrixAndVectorGPU` : Adds a Vector to every row of a Matrix
* `addScalarMatrixGPU` : Adds a Scalar to every element of a Matrix
* `addScalarVectorGPU` : Adds a Scalar to every element of a Vector
* `addScalar3DGPU` : Adds a Scalar to every element of a 3D Tensor
* `addBiasToFeatureMapGPU` : Adds bias values to a feature map Matrix
* `addBiasToMatMulOutGPU` : Adds a bias Vector to the output of a matrix multiplication
* `broadcastAddVectorToMatrixGPU` : Broadcasts and adds a Vector to a Matrix
* `scaleMatrixGPU` : Multiplies every element of a Matrix by a scalar value

**Activations:**
* `reluGPU` : Applies the ReLU activation function to a Vector
* `reluMatrixGPU` : Applies the ReLU activation function to a Matrix
* `sigmoidScalarGPU` : Applies the sigmoid function to a Scalar
* `sigmoidGPU` : Applies the sigmoid function to a Vector
* `sigmoidMatrixGPU` : Applies the sigmoid function to a Matrix
* `sigmoid3DGPU` : Applies the sigmoid function to a 3D Tensor
* `vectorTanhGPU` : Applies the tanh function to a Vector
* `tanhMatrixGPU` : Applies the tanh function to a Matrix
* `tanh3DGPU` : Applies the tanh function to a 3D Tensor
* `geluGPU` : Applies the GELU activation function to a Vector
* `geluMatrixGPU` : Applies the GELU activation function to a Matrix
* `softmaxMatrixGPU` : Applies the softmax function to a Matrix (turning it into probabilities)

**Loss Functions:**
* `binaryCrossEntropyGPU` : Calculates the binary cross-entropy loss between predictions and targets
* `mseGPU` : Calculates the Mean Squared Error between two Vectors
* `mseMatrixGPU` : Calculates the Mean Squared Error between two Matrices

**Optimizers:**
* `sgdUpdateGPU` : Updates parameters using Stochastic Gradient Descent
* `adamUpdateGPU` : Updates parameters using the Adam optimizer with momentum, variance, learning rate, and weight decay
* `clipGradValueGPU` : Clips gradient values to prevent exploding gradients

**Reductions:**
* `sumGPU` : Sums all elements in a Vector into a single Scalar
* `sumMatrixGPU` : Sums all elements in a Matrix into a single Scalar
* `embeddingLookupGPU` : Looks up embeddings from a weight Matrix using indices
* `embeddingLookupBatchGPU` : Looks up embeddings in batch mode from a weight Matrix
* `sumReduceColumnsGPU` : Sums each column of a Matrix into a Vector
* `sumReduceRowsGPU` : Sums each row of a Matrix into a Vector

**Tensor Manipulations:**
* `sliceColumnGPU` : Slices a range of columns from a Matrix
* `selectRowGPU` : Selects a specific row from a Matrix
* `selectMatrixFrom3DGPU` : Selects a specific Matrix from a 3D Tensor by index
* `concatenateGPU` : Concatenates two Vectors together
* `concatenateMatricesByColumnGPU` : Concatenates multiple Matrices together by columns
* `concatenate3DGPU` : Concatenates two 3D Tensors together
* `stackMatricesGPU` : Stacks multiple Matrices into a 3D Tensor
* `scatterHeadsGPU` : Scatters attention heads back into a single Matrix
* `padMatrixGPU` : Adds padding borders to a Matrix

**Advanced Layers:**
* `conv2dMultiChannelGPU` : Applies a multi-channel 2D convolution with kernel, bias, stride and padding options
* `conv2dSimpleGPU` : Applies a simple 2D convolution kernel to an input Matrix
* `im2colGPU` : Converts an image tensor to column format for efficient convolution
* `maxPool1dGPU` : Applies 1D max pooling with specified pool size and stride
* `maxPool2dGPU` : Applies 2D max pooling with specified pool size and stride
* `avgPool2dGPU` : Applies 2D average pooling with specified pool size and stride
* `globalAveragePoolingGPU` : Applies global average pooling over an entire feature map
* `batchNorm1dGPU` : Normalizes a Vector with batch normalization (gamma, beta, running statistics)
* `batchNorm2dGPU` : Normalizes a 3D Tensor with 2D batch normalization
* `layerNormMatrixGPU` : Applies layer normalization to a Matrix with learnable gamma and beta parameters
* `dropoutGPU` : Applies dropout by randomly setting elements to zero at a given rate
* `buildMarkovTableGPU` : Builds a Markov transition probability table from a sequence
* `markovPredictGPU` : Predicts next states using a Markov probability table

**Fused Kernels:**
* `matMulBiasReluGPU` : Fused operation combining matrix multiplication, bias addition, and ReLU activation
* `dotProductGPU` : Calculates the scalar dot product of two Vectors
* `l2NormGPU` : Calculates the L2 norm (Euclidean length) of a Vector
* `euclideanDistanceGPU` : Calculates the Euclidean distance between two Vectors
* `cosineSimilarityGPU` : Calculates the cosine similarity between two Vectors
* `maeLossGPU` : Calculates the Mean Absolute Error loss between predictions and targets

#### How to use:

```dart
import 'package:flutter_ml/full_library.dart';

void main() async{
  //  Initialize Cuda Engine and an Execution Tape
  await CudaEngine.initialize(debug: false);
  CommandBuffer tape=CommandBuffer();
  
  //  Initialize GPUTensors
  GPUTensor<Vector>VecA=GPUTensor([1.1, 2.2, 3.3]);
  GPUTensor<Vector>VecB=GPUTensor([4.4, 5.5, 6.6]);
  GPUTensor<Vector>VecC=GPUTensor([1.0,0.1,-1.0]);

  //  Calculation steps
  GPUTensor<Vector>VecD=addVectorGPU(VecA, VecB, tape);
  GPUTensor<Vector>VecE=elementWiseMultiplyGPU(VecC, VecD, tape);

  //  Execute the recorded operation
  CudaEngine.run(tape.bytes());


  VecE.toCpu();
  print("Result: ${VecE.value}");
  VecE.printGraph();
  
  //  Additional helper to trace exactly what is being sent to the backend
  TapeDecoder(tape.bytes()).decode();
}
```
This results in:
```
FFI: Creating new executor in SILENT mode...

 [5.5, 0.7699999809265137, -9.899999618530273]
 
🚀 GPU Computational Graph:
└──  t_gpu_4 [3] [GPU] (Op: elementWiseMultiplyGPU)
    ├──  t_gpu_2 [3] [GPU] (Leaf: VRAM Input)
    └──  t_gpu_3 [3] [GPU] (Op: addVectorGPU)
        ├──  t_gpu_0 [3] [GPU] (Leaf: VRAM Input)
        └──  t_gpu_1 [3] [GPU] (Leaf: VRAM Input)
        
📜 --- Decoding Execution Tape (62 bytes) ---
ℹ️ OP_ADD: t_gpu_3 = t_gpu_0 op t_gpu_1
ℹ️ OP_MULTIPLY: t_gpu_4 = t_gpu_2 op t_gpu_3
📜 --- End of Tape ---
```

#### Benchmark Speed on consumer hardware
To test the acceleration and speed of different mathematical operations on standard hardware,
a standardized test was done on an RTX 3060 12GB version with CUDA 12.1 and tensor core acceleration.
In between runs memory was wiped and the entire building process repeated, to give an estimation how much
cold loading time for each building process is to be expected, which for inference and training only has to be done once.

```
==================================================================
                 CUDA ENGINE PERFORMANCE BENCHMARK                
==================================================================
Vector Size: 134217728 elements (~537 MB)
Matrix Size: 8192x8192 elements
Iterations:  50
Note: VRAM is aggressively wiped and reallocated between each run to test loading speeds of different operations. 
------------------------------------------------------------------

[BENCHMARK] ADD        | Time:   5.24 ms | Bandwidth:   307.31 GB/s | Compute:   0.0256 TFLOPs Overhead  | Alloc/Tape: 2135.94 ms | Free:  38.08 ms
[BENCHMARK] MARKOV_TBL | Time:  11.98 ms | Bandwidth:    22.40 GB/s | Compute:   0.0056 TFLOPs Overhead  | Alloc/Tape: 648.10 ms  | Free:  16.40 ms
[BENCHMARK] MARKOV_PRD | Time:   3.38 ms | Bandwidth:   213.20 GB/s | Compute:   0.0474 TFLOPs Overhead  | Alloc/Tape: 1271.30 ms | Free:  26.51 ms
[BENCHMARK] SUBTRACT   | Time:   5.23 ms | Bandwidth:   308.12 GB/s | Compute:   0.0257 TFLOPs Overhead  | Alloc/Tape: 2256.46 ms | Free:  42.87 ms
[BENCHMARK] MULTIPLY   | Time:   5.14 ms | Bandwidth:   313.05 GB/s | Compute:   0.0261 TFLOPs Overhead  | Alloc/Tape: 2466.38 ms | Free:  15.88 ms
[BENCHMARK] DIVIDE     | Time:   5.39 ms | Bandwidth:   298.92 GB/s | Compute:   0.0249 TFLOPs Overhead  | Alloc/Tape: 2129.76 ms | Free:  48.59 ms
[BENCHMARK] ABS        | Time:   3.60 ms | Bandwidth:   298.63 GB/s | Compute:   0.0373 TFLOPs Overhead  | Alloc/Tape: 1704.05 ms | Free:  33.45 ms
[BENCHMARK] SQRT       | Time:   3.61 ms | Bandwidth:   297.78 GB/s | Compute:   0.0372 TFLOPs Overhead  | Alloc/Tape: 1794.70 ms | Free:  35.15 ms
[BENCHMARK] LOG        | Time:   4.71 ms | Bandwidth:   227.73 GB/s | Compute:   0.0285 TFLOPs Overhead  | Alloc/Tape: 2335.28 ms | Free:  33.97 ms
[BENCHMARK] POW        | Time:   3.63 ms | Bandwidth:   295.84 GB/s | Compute:   0.0370 TFLOPs Overhead  | Alloc/Tape: 1488.77 ms | Free:  30.05 ms
[BENCHMARK] CLAMP      | Time:   3.66 ms | Bandwidth:   293.12 GB/s | Compute:   0.0366 TFLOPs Overhead  | Alloc/Tape: 1673.10 ms | Free:  31.99 ms
[BENCHMARK] MATMUL     | Time:  81.23 ms | Bandwidth:     9.91 GB/s | Compute:  13.5354 TFLOPs Overhead  | Alloc/Tape: 1373.70 ms | Free:  17.80 ms
[BENCHMARK] TRANSPOSE  | Time:   2.38 ms | Bandwidth:   225.77 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 1059.03 ms | Free:  24.71 ms
[BENCHMARK] MAT_VEC    | Time:   0.87 ms | Bandwidth:   307.53 GB/s | Compute:   0.1537 TFLOPs Overhead  | Alloc/Tape: 379.82 ms  | Free:   5.80 ms
[BENCHMARK] ADD_BIAS   | Time:   1.93 ms | Bandwidth:   277.69 GB/s | Compute:   0.0347 TFLOPs Overhead  | Alloc/Tape: 710.38 ms  | Free:  12.69 ms
[BENCHMARK] SCALE_MAT  | Time:   1.99 ms | Bandwidth:   270.11 GB/s | Compute:   0.0338 TFLOPs Overhead  | Alloc/Tape: 1107.48 ms | Free:  17.39 ms
[BENCHMARK] ADD_SCALAR | Time:   3.53 ms | Bandwidth:   303.96 GB/s | Compute:   0.0380 TFLOPs Overhead  | Alloc/Tape: 1708.55 ms | Free:  19.71 ms
[BENCHMARK] RELU       | Time:   4.19 ms | Bandwidth:   256.38 GB/s | Compute:   0.0320 TFLOPs Overhead  | Alloc/Tape: 2446.01 ms | Free:  28.16 ms
[BENCHMARK] SIGMOID    | Time:   4.87 ms | Bandwidth:   220.38 GB/s | Compute:   0.0826 TFLOPs Overhead  | Alloc/Tape: 2385.56 ms | Free:  12.14 ms
[BENCHMARK] TANH       | Time:   4.46 ms | Bandwidth:   240.48 GB/s | Compute:   0.0902 TFLOPs Overhead  | Alloc/Tape: 2415.69 ms | Free:  11.53 ms
[BENCHMARK] GELU       | Time:   5.33 ms | Bandwidth:   201.43 GB/s | Compute:   0.1259 TFLOPs Overhead  | Alloc/Tape: 3790.56 ms | Free:  32.06 ms
[BENCHMARK] SOFTMAX    | Time:   3.12 ms | Bandwidth:   258.28 GB/s | Compute:   0.0646 TFLOPs Overhead  | Alloc/Tape: 1023.32 ms | Free:   8.64 ms
[BENCHMARK] BCE_LOSS   | Time:  24.06 ms | Bandwidth:    44.64 GB/s | Compute:   0.0223 TFLOPs Overhead  | Alloc/Tape: 1405.21 ms | Free:  12.10 ms
[BENCHMARK] MSE_VEC    | Time:  24.40 ms | Bandwidth:    44.01 GB/s | Compute:   0.0165 TFLOPs Overhead  | Alloc/Tape: 1410.52 ms | Free:  10.48 ms
[BENCHMARK] MSE_MAT    | Time:  12.11 ms | Bandwidth:    44.33 GB/s | Compute:   0.0166 TFLOPs Overhead  | Alloc/Tape: 711.45 ms  | Free:   5.07 ms
[BENCHMARK] SUM_VEC    | Time:   1.82 ms | Bandwidth:   294.55 GB/s | Compute:   0.0736 TFLOPs Overhead  | Alloc/Tape: 701.77 ms  | Free:  17.40 ms
[BENCHMARK] SUM_COLS   | Time:   0.89 ms | Bandwidth:   300.85 GB/s | Compute:   0.0752 TFLOPs Overhead  | Alloc/Tape: 523.13 ms  | Free:   7.02 ms
[BENCHMARK] SUM_ROWS   | Time:   7.13 ms | Bandwidth:    37.65 GB/s | Compute:   0.0094 TFLOPs Overhead  | Alloc/Tape: 349.99 ms  | Free:  16.54 ms
[BENCHMARK] EMBED_VEC  | Time:  15.30 ms | Bandwidth:   421.33 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 5286.45 ms | Free:  84.81 ms
[BENCHMARK] EMBED_MAT  | Time:  15.26 ms | Bandwidth:   422.36 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 4544.87 ms | Free:  88.81 ms
[BENCHMARK] SLICE_COL  | Time:   1.00 ms | Bandwidth:   269.41 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 962.99 ms  | Free:  14.19 ms
[BENCHMARK] SLICE_ROW  | Time:   0.11 ms | Bandwidth:    76.73 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 1174.23 ms | Free:   5.80 ms
[BENCHMARK] SLICE_3D   | Time:   0.08 ms | Bandwidth:    99.27 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 370.54 ms  | Free:   3.04 ms
[BENCHMARK] CONCAT_VEC | Time:   5.39 ms | Bandwidth:   199.13 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 2857.50 ms | Free:  30.98 ms
[BENCHMARK] STACK_MAT  | Time:   0.57 ms | Bandwidth:   235.83 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 224.28 ms  | Free:   3.67 ms
[BENCHMARK] SCAT_HEADS | Time:   0.33 ms | Bandwidth:    75.67 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape:  36.79 ms  | Free:   1.72 ms
[BENCHMARK] PAD_2D     | Time:   0.47 ms | Bandwidth:   278.27 GB/s | Compute:   0.0000 TFLOPs Overhead  | Alloc/Tape: 194.99 ms  | Free:   0.97 ms
[BENCHMARK] LAYER_NORM | Time:   3.47 ms | Bandwidth:   154.71 GB/s | Compute:   0.1547 TFLOPs Overhead  | Alloc/Tape: 727.34 ms  | Free:   5.75 ms
[BENCHMARK] DROPOUT    | Time:   2.75 ms | Bandwidth:   293.00 GB/s | Compute:   0.0244 TFLOPs Overhead  | Alloc/Tape: 1073.25 ms | Free:  18.37 ms
[BENCHMARK] DOT_PROD   | Time:   7.31 ms | Bandwidth:   293.90 GB/s | Compute:   0.0367 TFLOPs Overhead  | Alloc/Tape: 2356.80 ms | Free:  32.84 ms
[BENCHMARK] L2_NORM    | Time:   5.66 ms | Bandwidth:   284.70 GB/s | Compute:   0.0475 TFLOPs Overhead  | Alloc/Tape: 1797.15 ms | Free:  14.86 ms
[BENCHMARK] EUC_DIST   | Time:  11.90 ms | Bandwidth:   270.64 GB/s | Compute:   0.0338 TFLOPs Overhead  | Alloc/Tape: 3751.22 ms | Free:  34.63 ms
[BENCHMARK] COS_SIM    | Time:  18.41 ms | Bandwidth:   291.55 GB/s | Compute:   0.0437 TFLOPs Overhead  | Alloc/Tape: 3780.90 ms | Free:  31.48 ms
[BENCHMARK] MAE_LOSS   | Time:  48.99 ms | Bandwidth:    65.75 GB/s | Compute:   0.0082 TFLOPs Overhead  | Alloc/Tape: 3310.34 ms | Free:  29.07 ms
FFI: Freeing executor...
```



## Future Plans:
1.  Integration of whole models directly into the engine
2.  Addition of science cores into the engine
3.  Addition of WebGPU to allow ML directly on device 

