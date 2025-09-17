#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/efficient_plus.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>

int main(int argc, char *argv[]) {
    // argv[1]: scan/compact
    // argv[2]: cpu/naive/efficient/efficient_plus/thrust
    // argv[3]: input size
    // argv[4]: block size
    // argv[5]: elements per thread

    if (argc != 6) {
        return -1;
    }
    const std::string operation(argv[1]);
    const std::string implementation(argv[2]);
    const int inputSize = std::stoi(argv[3]);
    const int blockSize = std::stoi(argv[4]);
    const int elementsPerThread = std::stoi(argv[5]);

    std::vector<int> inputs(inputSize);
    std::vector<int> outputs(inputSize);
    {
        std::random_device device;
        std::mt19937 generator(device());
        int maxValue;
        if (operation == "scan") {
            maxValue = 50 - 1;
        } else if (operation == "compact") {
            maxValue = 4 - 1;
        } else {
            return -1;
        }
        std::uniform_int_distribution<int> distribution(0, maxValue);
        for (auto &i : inputs) {
            i = distribution(generator);
        }
    }

    const auto isGpu{implementation != "cpu"};
    StreamCompaction::Common::PerformanceTimer *timer = nullptr;
    if (implementation == "cpu") {
        timer = &StreamCompaction::CPU::timer();
    } else if (implementation == "naive") {
        timer = &StreamCompaction::Naive::timer();
    } else if (implementation == "efficient") {
        timer = &StreamCompaction::Efficient::timer();
    } else if (implementation == "efficient_plus") {
        timer = &StreamCompaction::EfficientPlus::timer();
    } else if (implementation == "thrust") {
        timer = &StreamCompaction::Thrust::timer();
    } else {
        return -1;
    }

    // Some implementations (e.g., Thrust) allocates memory internally on the first run, so we run
    // the operation twice and only measure the later one.
#define REPEAT_TWICE(function, ...)                                                                \
    do {                                                                                           \
        function(inputSize, outputs.data(), inputs.data(), __VA_ARGS__);                           \
        std::this_thread::sleep_for(std::chrono::milliseconds(10));                                \
        function(inputSize, outputs.data(), inputs.data(), __VA_ARGS__);                           \
    } while (false)

    if (operation == "scan") {
        if (implementation == "cpu") {
            REPEAT_TWICE(StreamCompaction::CPU::scan);
        } else if (implementation == "naive") {
            REPEAT_TWICE(StreamCompaction::Naive::scan, blockSize);
        } else if (implementation == "efficient") {
            REPEAT_TWICE(StreamCompaction::Efficient::scan, blockSize);
        } else if (implementation == "efficient_plus") {
            REPEAT_TWICE(StreamCompaction::EfficientPlus::scan, blockSize, elementsPerThread);
        } else if (implementation == "thrust") {
            REPEAT_TWICE(StreamCompaction::Thrust::scan);
        } else {
            return -1;
        }
    } else if (operation == "compact") {
        REPEAT_TWICE(StreamCompaction::Efficient::compact, blockSize);
    } else {
        return -1;
    }

#undef REPEAT_TWICE

    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (isGpu) {
        std::cout << timer->getGpuElapsedTimeForPreviousOperation();
    } else {
        std::cout << timer->getCpuElapsedTimeForPreviousOperation();
    }
    std::cout << std::endl;
    return 0;
}
