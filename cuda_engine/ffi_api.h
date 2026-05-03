#ifndef FFI_API_H
#define FFI_API_H

#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
    #define API_EXPORT __declspec(dllexport)
#else
    #define API_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
///                        Lifecycle Management                               //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT int64_t create_executor(bool debug);
API_EXPORT void free_executor(int64_t executor_ptr);
API_EXPORT void free_tensor(int64_t executor_ptr, const char* name);

////////////////////////////////////////////////////////////////////////////////
///                        Data Transfer (Host <-> VRAM)                      //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void free_host_buffer(float* buffer);
API_EXPORT void init_random_uniform(int64_t executor_ptr, const char* name, float scale, int seed);
API_EXPORT void load_tensor_h2d(int64_t executor_ptr, const char* name, float* data, int num_dims, int* shape_ptr);
API_EXPORT float* retrieve_tensor_d2h(int64_t executor_ptr, const char* name, int* out_size);
API_EXPORT void retrieve_tensor_d2h_into(int64_t executor_ptr, const char* name, float* dest_buffer);

////////////////////////////////////////////////////////////////////////////////
///                       Pinned Memory Management (DMA)                      //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT float* allocate_pinned_memory(int total_elements);
API_EXPORT void free_pinned_memory(float* pinned_ptr);
API_EXPORT void load_tensor_from_pinned(int64_t executor_ptr, const char* name, float* pinned_ptr, int num_dims, int* shape_ptr);

////////////////////////////////////////////////////////////////////////////////
///                        Engine Execution                                   //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void run_tape(int64_t executor_ptr, const uint8_t* tape, int total_bytes);

////////////////////////////////////////////////////////////////////////////////
///                        Utilities & Introspection                          //
////////////////////////////////////////////////////////////////////////////////

API_EXPORT void add_pointers(float* dest, const float* src, int length);
API_EXPORT void free_string(const char* str);
API_EXPORT int32_t get_tensor_count(int64_t executor_ptr);
API_EXPORT const char* get_tensor_names(int64_t executor_ptr);
API_EXPORT void print_tensor_registry(int64_t executor_ptr);

#ifdef __cplusplus
}
#endif

#endif // FFI_API_H