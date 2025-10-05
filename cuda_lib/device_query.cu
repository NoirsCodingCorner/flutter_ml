#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount failed!" << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available CUDA-enabled devices." << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
    }

    return 0;
}