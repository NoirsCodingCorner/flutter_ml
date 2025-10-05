import 'dart:ffi';
import 'package:ffi/ffi.dart';

// FFI function signatures are defined at the top level
typedef CMultiplyMatrices = Void Function(Pointer<Float> a, Pointer<Float> b,
    Pointer<Float> c, Int32 m, Int32 k, Int32 n);

typedef DartMultiplyMatrices = void Function(Pointer<Float> a, Pointer<Float> b,
    Pointer<Float> c, int m, int k, int n);

/// A utility class for calling CUDA functions.
class CudaCaller {
  // This static final variable holds the loaded FFI function.
  // It's initialized lazily and only once when the class is first accessed.
  static final DartMultiplyMatrices _multiplyMatrices =
  DynamicLibrary.open('cuda_lib/cuda_lib.dll')
      .lookup<NativeFunction<CMultiplyMatrices>>('multiplyMatrices')
      .asFunction();

  /// Multiplies matrix A (m x k) by matrix B (k x n) on the GPU.
  ///
  /// This is a static method, so you can call it directly:
  /// `CudaCaller.matmult(...)`
  static List<double> matmult(
      List<double> matA, List<double> matB, int m, int k, int n) {
    // Check if the input lists have the correct number of elements
    assert(matA.length == m * k, 'Matrix A has incorrect dimensions.');
    assert(matB.length == k * n, 'Matrix B has incorrect dimensions.');

    // Allocate native memory for all three matrices
    final Pointer<Float> pA = calloc<Float>(m * k);
    final Pointer<Float> pB = calloc<Float>(k * n);
    final Pointer<Float> pC = calloc<Float>(m * n);

    try {
      // Copy input data from Dart lists to native memory
      for (int i = 0; i < matA.length; i++) {
        pA[i] = matA[i];
      }
      for (int i = 0; i < matB.length; i++) {
        pB[i] = matB[i];
      }

      // Call the static GPU function pointer
      _multiplyMatrices(pA, pB, pC, m, k, n);

      // Copy the result from native memory back to a new Dart list
      final List<double> result = List<double>.filled(m * n, 0);
      for (int i = 0; i < result.length; i++) {
        result[i] = pC[i];
      }
      return result;
    } finally {
      // CRITICAL: Ensure native memory is always freed.
      calloc.free(pA);
      calloc.free(pB);
      calloc.free(pC);
    }
  }
}

// Helper function to print a matrix for verification
void printMatrix(List<double> mat, int rows, int cols, String name) {
  print("\nMatrix $name ($rows x $cols):");
  for (int i = 0; i < rows; i++) {
    List<double> row = mat.sublist(i * cols, (i + 1) * cols);
    String rowStr = row.map((e) => e.toStringAsFixed(1).padLeft(5)).join(" ");
    print("  [ $rowStr ]");
  }
}

void main() {
  // Define the matrices and their dimensions.
  // C (m x n) = A (m x k) * B (k x n)
  int m = 2;
  int k = 3;
  int n = 4;

  List<double> matrixA = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
  List<double> matrixB = [
    7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0,
    15.0, 16.0, 17.0, 18.0
  ]; // 3x4

  printMatrix(matrixA, m, k, "A");
  printMatrix(matrixB, k, n, "B");

  // Call the static method directly on the class.
  // No instance is needed.
  print('\n🚀 Calling GPU to multiply matrices...');
  final List<double> matrixC = CudaCaller.matmult(matrixA, matrixB, m, k, n);
  print('...Done.');

  // Print the result.
  printMatrix(matrixC, m, n, "C (Result)");
}