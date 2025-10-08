import 'dart:ffi';
import 'package:ffi/ffi.dart'; // For calloc

// 1. Define the FFI signature for your C++ multiplyMatrices function.
typedef CMultiplyMatrices = Void Function(
    Pointer<Float> a, Pointer<Float> b, Pointer<Float> c,
    Int32 m, Int32 k, Int32 n);

typedef DartMultiplyMatrices = void Function(
    Pointer<Float> a, Pointer<Float> b, Pointer<Float> c,
    int m, int k, int n);

void main() {
  // 2. Load your CUDA library.
  final DynamicLibrary dylib = DynamicLibrary.open('cuda_lib/cuda_lib.dll');

  // 3. Look up the multiplyMatrices function.
  final DartMultiplyMatrices multiplyMatrices = dylib
      .lookup<NativeFunction<CMultiplyMatrices>>('multiplyMatrices')
      .asFunction();

  print('✅ Successfully loaded the multiplyMatrices function.');

  // --- Benchmark Setup ---
  final int size = 1024; // Matrix size (e.g., 1024x1024)
  final int m = size, k = size, n = size;
  final int numElements = size * size;

  // 4. Allocate memory for the matrices on the Dart side.
  // This memory is accessible by the C++ code via pointers.
  final Pointer<Float> a = calloc<Float>(numElements);
  final Pointer<Float> b = calloc<Float>(numElements);
  final Pointer<Float> c = calloc<Float>(numElements);

  // Initialize input matrices with dummy data (e.g., all 1.0s).
  for (int i = 0; i < numElements; i++) {
    a[i] = 1.0;
    b[i] = 1.0;
  }

  print('--- Timing a single $size x $size matrix multiplication round trip ---');

  // 5. Start the timer.
  final Stopwatch stopwatch = Stopwatch()..start();

  // 6. Call the C++ function.
  // This is the entire "ping-pong" trip:
  // Dart -> C++ -> GPU -> C++ -> Dart
  multiplyMatrices(a, b, c, m, k, n);

  // 7. Stop the timer.
  stopwatch.stop();

  // 8. Print the result.
  print('✅ Round trip finished!');
  print('Total time: ${stopwatch.elapsedMilliseconds} ms');

  // (Optional) Verify a result to make sure it worked.
  // For two matrices of all 1s, each element in the result should be 'k'.
  print('Verification: c[0] = ${c[0]} (should be $k)');

  // 9. IMPORTANT: Free the allocated memory.
  calloc.free(a);
  calloc.free(b);
  calloc.free(c);
}