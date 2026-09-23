#include "ffi_api.h"
#include "manager.h"
#include "graph_executor.h"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////
///                        Internal Structures                                //
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Wrapper struct to manage the lifecycle of the CudaManager and GraphExecutor.
 * This ensures that when we free the executor, both the manager (memory) and
 * the executor (logic) are destroyed in the correct order.
 */
struct ExecutorHandle {
    CudaManager* manager;
    GraphExecutor* executor;

    ExecutorHandle(bool debug = false) {
        try {
            manager = new CudaManager();
            executor = new GraphExecutor(*manager, debug);
        } catch (const std::exception& e) {
            std::cerr << "FFI: Failed to initialize CudaManager: " << e.what() << std::endl;
            manager = nullptr;
            executor = nullptr;
        }
    }

    ~ExecutorHandle() {
        if (executor) delete executor;
        if (manager) delete manager;
    }
};

////////////////////////////////////////////////////////////////////////////////
///                        Lifecycle Management                               //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT int64_t create_executor(bool debug) {
    if (debug) {
        std::cout << "FFI: Creating new executor in DEBUG mode..." << std::endl;
    } else {
        std::cout << "FFI: Creating new executor in SILENT mode..." << std::endl;
    }
    ExecutorHandle* handle = new ExecutorHandle(debug);
    return reinterpret_cast<int64_t>(handle);
}

API_EXPORT void free_executor(int64_t executor_ptr) {
    std::cout << "FFI: Freeing executor..." << std::endl;
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (handle) {
        delete handle;
    }
}

API_EXPORT void free_tensor(int64_t executor_ptr, const char* name) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager || !name) return;

    try {
        std::string tensor_name(name);
        if (handle->manager->exists(tensor_name)) {
            handle->manager->free(tensor_name);
        }
    } catch (const std::exception& e) {
        std::cerr << "FFI Error in free_tensor(" << name << "): " << e.what() << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
///                        Data Transfer (Host <-> VRAM)                      //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void free_host_buffer(float* buffer) {
    if (buffer) {
        cudaFreeHost(buffer);
    }
}

API_EXPORT void init_random_uniform(int64_t executor_ptr, const char* name, float scale, int seed) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (handle && handle->manager) {
        handle->manager->init_random_uniform(name, scale, seed);
    }
}

API_EXPORT void load_tensor_h2d(int64_t executor_ptr, const char* name, float* data, int num_dims, int* shape_ptr) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager || !name || !data || !shape_ptr) return;

    std::string tensor_name(name);
    std::vector<int> new_shape(shape_ptr, shape_ptr + num_dims);

    try {
        if (handle->manager->exists(tensor_name)) {
            const Gpu_Element& existing_element = handle->manager->get_element(tensor_name);
            bool shape_mismatch = false;

            if (existing_element.sizes.size() != new_shape.size()) {
                shape_mismatch = true;
            } else {
                for (size_t i = 0; i < new_shape.size(); ++i) {
                    if (existing_element.sizes[i] != new_shape[i]) {
                        shape_mismatch = true;
                        break;
                    }
                }
            }

            if (shape_mismatch) {
                handle->manager->free(tensor_name);
                handle->manager->allocate(tensor_name, new_shape);
            }
        } else {
            handle->manager->allocate(tensor_name, new_shape);
        }

        handle->manager->copyHostToDevice(tensor_name, data);

    } catch (const std::exception& e) {
        std::cerr << "FFI Error in load_tensor_h2d(" << name << "): " << e.what() << std::endl;
    }
}

API_EXPORT float* retrieve_tensor_d2h(int64_t executor_ptr, const char* name, int* out_size) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager || !name || !out_size) { *out_size = 0; return nullptr; }

    std::string tensor_name(name);

    try {
        const Gpu_Element& elem = handle->manager->get_element(tensor_name);

        int total_elements = 1;
        for(size_t i = 0; i < elem.sizes.size(); ++i) {
            total_elements *= elem.sizes[i];
        }
        *out_size = total_elements;

        float* host_buffer = nullptr;
        cudaError_t err = cudaMallocHost((void**)&host_buffer, total_elements * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "FFI: cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
            *out_size = 0;
            return nullptr;
        }

        cudaMemcpy(host_buffer, elem.ptr, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

        return host_buffer;

    } catch (const std::exception& e) {
        std::cerr << "FFI Error in retrieve_tensor_d2h(" << name << "): " << e.what() << std::endl;
        *out_size = 0;
        return nullptr;
    }
}

API_EXPORT void retrieve_tensor_d2h_into(int64_t executor_ptr, const char* name, float* dest_buffer) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager || !name || !dest_buffer) return;

    std::string tensor_name(name);

    try {
        const Gpu_Element& elem = handle->manager->get_element(tensor_name);

        int total_elements = 1;
        for(size_t i = 0; i < elem.sizes.size(); ++i) {
            total_elements *= elem.sizes[i];
        }

        cudaMemcpy(dest_buffer, elem.ptr, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    } catch (const std::exception& e) {
        std::cerr << "FFI Error in retrieve_tensor_d2h_into(" << name << "): " << e.what() << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
///                       Pinned Memory Management (DMA)                      //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT float* allocate_pinned_memory(int total_elements) {
    float* pinned_ptr = nullptr;
    cudaError_t err = cudaMallocHost((void**)&pinned_ptr, total_elements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "FFI Error: Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return pinned_ptr;
}

API_EXPORT void free_pinned_memory(float* pinned_ptr) {
    if (pinned_ptr) {
        cudaFreeHost(pinned_ptr);
    }
}

API_EXPORT void load_tensor_from_pinned(int64_t executor_ptr, const char* name, float* pinned_ptr, int num_dims, int* shape_ptr) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager || !name || !pinned_ptr || !shape_ptr) return;

    std::string tensor_name(name);
    std::vector<int> new_shape(shape_ptr, shape_ptr + num_dims);

    try {
        if (handle->manager->exists(tensor_name)) {
            const Gpu_Element& existing_element = handle->manager->get_element(tensor_name);
            bool shape_mismatch = false;
            if (existing_element.sizes.size() != new_shape.size()) {
                shape_mismatch = true;
            } else {
                for (size_t i = 0; i < new_shape.size(); ++i) {
                    if (existing_element.sizes[i] != new_shape[i]) {
                        shape_mismatch = true;
                        break;
                    }
                }
            }
            if (shape_mismatch) {
                handle->manager->free(tensor_name);
                handle->manager->allocate(tensor_name, new_shape);
            }
        } else {
            handle->manager->allocate(tensor_name, new_shape);
        }

        int total_elements = 1;
        for (int dim : new_shape) {
            total_elements *= dim;
        }

        const Gpu_Element& elem = handle->manager->get_element(tensor_name);
        cudaMemcpy(elem.ptr, pinned_ptr, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    } catch (const std::exception& e) {
        std::cerr << "FFI Error in load_tensor_from_pinned(" << name << "): " << e.what() << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
///                        Engine Execution                                   //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void run_tape(int64_t executor_ptr, const uint8_t* tape, int total_bytes) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->executor || !tape) return;

    try {
        handle->executor->run_tape(tape, total_bytes);
    } catch (const std::exception& e) {
        std::cerr << "FFI Error in run_tape: " << e.what() << " : " << total_bytes << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
///                        Utilities & Introspection                          //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void add_pointers(float* dest, const float* src, int length) {
    for (int i = 0; i < length; i++) {
        dest[i] += src[i];
    }
}

API_EXPORT void free_string(const char* str) {
    if (str) {
        delete[] str;
    }
}

API_EXPORT int32_t get_tensor_count(int64_t executor_ptr) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager) return -1;
    return static_cast<int32_t>(handle->manager->memory.size());
}

API_EXPORT const char* get_tensor_names(int64_t executor_ptr) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager) return nullptr;

    std::string result = "";
    std::map<std::string, Gpu_Element>::iterator it;

    for (it = handle->manager->memory.begin(); it != handle->manager->memory.end(); it++) {
        if (!result.empty()) {
            result += ",";
        }
        result += it->first;
    }

    char* c_str = new char[result.length() + 1];
    std::strcpy(c_str, result.c_str());

    return c_str;
}

API_EXPORT void print_tensor_registry(int64_t executor_ptr) {
    ExecutorHandle* handle = reinterpret_cast<ExecutorHandle*>(executor_ptr);
    if (!handle || !handle->manager) return;
    std::cout << "=== Tensor Registry (" << handle->manager->memory.size() << " tensors) ===" << std::endl;
    size_t total_bytes = 0;
    for (auto& it : handle->manager->memory) {
        int elems = 1;
        for (int s : it.second.sizes) elems *= s;
        size_t bytes = elems * sizeof(float);
        total_bytes += bytes;
        std::cout << "  [" << it.first << "] " << elems << " floats (" << bytes / 1024 << " KB)" << std::endl;
    }
    std::cout << "  Total: " << total_bytes / 1024 << " KB" << std::endl;
}