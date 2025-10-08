## Features

Full autograd engine in pure dart code. Includes full control over all the math and allows for rapid development of new 
models, architectures etc.

## Getting started

This is a learning package. To start id recommend you to look in to the tensor class itself as it contains the most interesting
aspects of the engine. It contains the bare bone building blocks of the engine itself and functions as all there is about gradient propagation.

To boot there are many additional on top building blocks that help you understand how this complex system works.
Math operations are open and can be looked it, the entire structure is open making it super easy to debug and experiment with.
Although this costs performance due to the obvious loss of parallelism it is a great way to make understanding the engine easier.

## Features

This package includes many features. The most important of them are: 
- Auto gradient engine
- Auto building of computational graphs
- General Layer structure similar to other frameworks such as Pytorch or tensorflow
- Ready to use activation functions
- Ready to use optimizers
- Ready to use loss functions
- Detailed Computational Graph printouts via a custom Logger
- Transformer networks integrated
- many more features...

```dart
const like = 'sample';
```

## Additional information

TODO: Tell users more about the package: where to find more information, how to
contribute to the package, how to file issues, what response they can expect
from the package authors, and more.
