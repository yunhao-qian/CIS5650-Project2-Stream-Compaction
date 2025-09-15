#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>

#include "testing_helpers.hpp"

constexpr int MAX_SIZE = 1 << 27;
std::vector<int> inputs(MAX_SIZE);
std::vector<int> outputs(MAX_SIZE);

void prepareData(int n) {
    genArray(n, inputs.data(), 50);
    zeroArray(n, outputs.data());
}

template <bool IsGpu, typename Timer, typename Func>
float measureTime(Timer &timer, Func func, int n) {
    const int warmup = 1;
    const int iterations = 10;
    for (int i = 0; i < warmup; i++) {
        prepareData(n);
        func();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    float totalTime = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        prepareData(n);
        func();
        if constexpr (IsGpu) {
            totalTime += timer.getGpuElapsedTimeForPreviousOperation();
        } else {
            totalTime += timer.getCpuElapsedTimeForPreviousOperation();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return totalTime / iterations;
}

int main() {
    std::ofstream outFile("performance_comparison.nljson");

    for (int exponent = 4; exponent <= 27; ++exponent) {
        const int baseN = 1 << exponent;
        for (int n : {baseN - 3, baseN}) {
            float cpuTime = measureTime<false>(
                StreamCompaction::CPU::timer(),
                [&, n] { StreamCompaction::CPU::scan(n, outputs.data(), inputs.data()); }, n);
            float naiveTime = measureTime<true>(
                StreamCompaction::Naive::timer(),
                [&, n]() { StreamCompaction::Naive::scan(n, outputs.data(), inputs.data()); }, n);
            float efficientTime = measureTime<true>(
                StreamCompaction::Efficient::timer(),
                [&, n]() { StreamCompaction::Efficient::scan(n, outputs.data(), inputs.data()); },
                n);
            float thrustTime = measureTime<true>(
                StreamCompaction::Thrust::timer(),
                [&, n]() { StreamCompaction::Thrust::scan(n, outputs.data(), inputs.data()); }, n);
            for (const auto [name, time] : {
                     std::pair("cpu", cpuTime),
                     std::pair("naive", naiveTime),
                     std::pair("efficient", efficientTime),
                     std::pair("thrust", thrustTime),
                 }) {
                outFile << "{ "
                        << "\"n\": " << n << ", "
                        << "\"method\": \"" << name << "\", "
                        << "\"time\": " << time << " }" << std::endl;
            }
            std::cout << "Completed n = " << n << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}
