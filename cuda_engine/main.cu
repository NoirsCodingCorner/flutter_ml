#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include "manager.h"
#include "graph_executor.h"

////////////////////////////////////////////////////////////////////////////////
///                     Dart Mock: Byte Packing Utilities                     //
////////////////////////////////////////////////////////////////////////////////

void write_int32(std::vector<uint8_t>& tape, int32_t val) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
    tape.insert(tape.end(), ptr, ptr + sizeof(int32_t));
}

void write_float(std::vector<uint8_t>& tape, float val) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
    tape.insert(tape.end(), ptr, ptr + sizeof(float));
}

void write_bool(std::vector<uint8_t>& tape, bool val) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
    tape.insert(tape.end(), ptr, ptr + sizeof(bool));
}

void write_string(std::vector<uint8_t>& tape, const std::string& str) {
    uint16_t len = static_cast<uint16_t>(str.length());
    const uint8_t* len_ptr = reinterpret_cast<const uint8_t*>(&len);
    tape.insert(tape.end(), len_ptr, len_ptr + sizeof(uint16_t));

    const uint8_t* str_ptr = reinterpret_cast<const uint8_t*>(str.data());
    tape.insert(tape.end(), str_ptr, str_ptr + len);
}

////////////////////////////////////////////////////////////////////////////////
///                            Benchmarking Loop                              //
////////////////////////////////////////////////////////////////////////////////

int main() {
    std::cout << "--- Neurosymbolic Engine: Sub-Microsecond Profiler ---" << std::endl;
    std::cout << std::left << std::setw(6) << "Size"
              << std::setw(18) << "Add Dispatch"
              << std::setw(18) << "Add Sync"
              << std::setw(18) << "MatMul Dispatch"
              << std::setw(18) << "MatMul Sync" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    CudaManager manager;
    GraphExecutor executor(manager, false);

    int N = 1;

    while (N <= 256) {
        std::vector<int> shape = {N, N};
        manager.allocate("A", shape);
        manager.allocate("B", shape);
        manager.allocate("C", shape);

        int iterations = 10000;

        // ---------------------------------------------------------
        // TEST 1: Elementwise Add Profiling
        // ---------------------------------------------------------
        std::vector<uint8_t> tape_add;
        write_int32(tape_add, OP_ADD);
        write_string(tape_add, "A");
        write_string(tape_add, "B");
        write_string(tape_add, "C");

        executor.run_tape(tape_add.data(), tape_add.size(), true); // Warmup

        double add_dispatch_total = 0.0;
        double add_sync_total = 0.0;

        for (int i = 0; i < iterations; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. CPU Phase: Decode tape and launch kernel into stream
            executor.run_tape(tape_add.data(), tape_add.size(), false);

            auto t1 = std::chrono::high_resolution_clock::now();

            // 2. GPU Phase: Block CPU until PCIe interrupt confirms completion
            cudaDeviceSynchronize();

            auto t2 = std::chrono::high_resolution_clock::now();

            add_dispatch_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
            add_sync_total += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }

        // ---------------------------------------------------------
        // TEST 2: MatMul Profiling
        // ---------------------------------------------------------
        std::vector<uint8_t> tape_matmul;
        write_int32(tape_matmul, OP_MATMUL);
        write_string(tape_matmul, "A");
        write_string(tape_matmul, "B");
        write_string(tape_matmul, "C");
        write_bool(tape_matmul, false);
        write_bool(tape_matmul, false);
        write_float(tape_matmul, 1.0f);
        write_float(tape_matmul, 0.0f);
        write_bool(tape_matmul, true);

        executor.run_tape(tape_matmul.data(), tape_matmul.size(), true); // Warmup

        double mul_dispatch_total = 0.0;
        double mul_sync_total = 0.0;

        for (int i = 0; i < iterations; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            executor.run_tape(tape_matmul.data(), tape_matmul.size(), false);

            auto t1 = std::chrono::high_resolution_clock::now();

            cudaDeviceSynchronize();

            auto t2 = std::chrono::high_resolution_clock::now();

            mul_dispatch_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
            mul_sync_total += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }

        // Print Averaged Results
        std::cout << std::left << std::setw(6) << N
                  << std::setw(18) << (std::to_string(add_dispatch_total / iterations).substr(0, 5) + " us")
                  << std::setw(18) << (std::to_string(add_sync_total / iterations).substr(0, 5) + " us")
                  << std::setw(18) << (std::to_string(mul_dispatch_total / iterations).substr(0, 5) + " us")
                  << std::setw(18) << (std::to_string(mul_sync_total / iterations).substr(0, 5) + " us")
                  << std::endl;

        manager.free("A");
        manager.free("B");
        manager.free("C");
        N *= 2;
    }

    return 0;
}