#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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

    if (operation == "scan") {
        if (implementation == "cpu") {
            if (blockSize >= 0 || elementsPerThread >= 0) {
                return -1;
            }
            StreamCompaction::CPU::scan(inputSize, outputs.data(), inputs.data());
        } else if (implementation == "naive") {
            if (elementsPerThread >= 0) {
                return -1;
            }
            if (blockSize >= 0) {
                StreamCompaction::Naive::scan(inputSize, outputs.data(), inputs.data(), blockSize);
            } else {
                StreamCompaction::Naive::scan(inputSize, outputs.data(), inputs.data());
            }
        } else if (implementation == "efficient") {
            if (elementsPerThread >= 0) {
                return -1;
            }
            if (blockSize >= 0) {
                StreamCompaction::Efficient::scan(inputSize, outputs.data(), inputs.data(),
                                                  blockSize);
            } else {
                StreamCompaction::Efficient::scan(inputSize, outputs.data(), inputs.data());
            }
        } else if (implementation == "efficient_plus") {
            if (blockSize >= 0) {
                if (elementsPerThread >= 0) {
                    StreamCompaction::EfficientPlus::scan(inputSize, outputs.data(), inputs.data(),
                                                          blockSize, elementsPerThread);
                } else {
                    StreamCompaction::EfficientPlus::scan(inputSize, outputs.data(), inputs.data(),
                                                          blockSize);
                }
            } else {
                if (elementsPerThread >= 0) {
                    return -1;
                } else {
                    StreamCompaction::EfficientPlus::scan(inputSize, outputs.data(), inputs.data());
                }
            }
        } else if (implementation == "thrust") {
            if (blockSize >= 0 || elementsPerThread >= 0) {
                return -1;
            }
            StreamCompaction::Thrust::scan(inputSize, outputs.data(), inputs.data());
        } else {
            return -1;
        }
    } else if (operation == "compact") {
        if (implementation == "efficient") {
            if (elementsPerThread >= 0) {
                return -1;
            }
            if (blockSize >= 0) {
                StreamCompaction::Efficient::compact(inputSize, outputs.data(), inputs.data(),
                                                     blockSize);
            } else {
                StreamCompaction::Efficient::compact(inputSize, outputs.data(), inputs.data());
            }
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (isGpu) {
        std::cout << timer->getGpuElapsedTimeForPreviousOperation() << std::endl;
    } else {
        std::cout << timer->getCpuElapsedTimeForPreviousOperation() << std::endl;
    }
    return 0;
}
