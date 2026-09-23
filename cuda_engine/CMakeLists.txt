cmake_minimum_required(VERSION 3.21)

project(cuda_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. THE HARDWARE FIX: Target multiple GPU Architectures (Fat Binary)
# 75 = RTX 20-series, 86 = RTX 30-series, 89 = RTX 40-series
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 89-virtual)

# 2. THE RUNTIME FIX: Statically link the base CUDA Runtime so users don't need the toolkit
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDAToolkit REQUIRED)

set(CXX_SOURCES
        graph_executor.cpp
        ffi_api.cpp
)

set(CUDA_SOURCES
        manager.cu
)

add_library(cuda_engine SHARED
        ${CXX_SOURCES}
        ${CUDA_SOURCES}
)

set_target_properties(cuda_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "cuda_executor"
)

# 3. Use the dynamic cuBLAS library
target_link_libraries(cuda_engine PRIVATE
        CUDA::cublas
)