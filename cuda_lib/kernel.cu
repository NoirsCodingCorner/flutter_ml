#include <iostream>
#include <cublas_v2.h>
#include <chrono> // Required for timing

// This is your existing function. No changes are needed here.
extern "C" __declspec(dllexport)
void multiplyMatrices(float* h_a, float* h_b, float* h_c, int m, int k, int n) {
    // A is an m x k matrix
    // B is a k x n matrix
    // C is the resulting m x n matrix

    float *d_a, *d_b, *d_c;
    int a_size = m * k * sizeof(float);
    int b_size = k * n * sizeof(float);
    int c_size = m * n * sizeof(float);

    // 1. Allocate memory on the GPU device
    cudaMalloc((void**)&d_a, a_size);
    cudaMalloc((void**)&d_b, b_size);
    cudaMalloc((void**)&d_c, c_size);

    // 2. Copy matrices from host (CPU) to device (GPU)
    cudaMemcpy(d_a, h_a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

    // 3. Use cuBLAS to perform the matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                d_b, n,
                d_a, k,
                &beta,
                d_c, n);

    cublasDestroy(handle);

    // 4. Copy the result matrix C from device (GPU) back to host (CPU)
    cudaMemcpy(h_c, d_c, c_size, cudaMemcpyDeviceToHost);

    // 5. Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


// --- NEW BENCHMARK FUNCTION ---

/// Runs an increasing benchmark for matrix multiplication and returns
/// the size at which the computation time exceeded 1000 milliseconds.
extern "C" __declspec(dllexport)
int benchtest() {
    int size = 100;
    const int step = 100;

    // Loop forever, increasing the size with each iteration
    for (;; size += step) {
        int m = size;
        int k = size;
        int n = size;

        // Allocate memory on the CPU (host) for the matrices
        float* h_a = new float[m * k];
        float* h_b = new float[k * n];
        float* h_c = new float[m * n];

        // Start the high-resolution timer
        auto start = std::chrono::high_resolution_clock::now();

        // Call the existing function to perform the GPU work
        multiplyMatrices(h_a, h_b, h_c, m, k, n);

        // Stop the timer and calculate the duration in milliseconds
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        long long elapsedTime = duration.count();

        // Print progress to the console
        std::cout << "Size: " << size << "x" << size << " ... Time: " << elapsedTime << " ms" << std::endl;

        // IMPORTANT: Free the host memory to avoid leaks
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;

        // Check if the time limit was exceeded
        if (elapsedTime > 1000) {
            return size; // Return the size that broke the 1-second barrier
        }
    }

    return 0; // This line should ideally not be reached
}