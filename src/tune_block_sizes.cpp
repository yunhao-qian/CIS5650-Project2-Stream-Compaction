#include <iostream>
#include <limits>
#include <vector>

#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>

#include "testing_helpers.hpp"

template <typename Timer, typename Func>
float measureTime(Timer &timer, Func func) {
    const int warmup = 10;
    const int iterations = 100;
    for (int i = 0; i < warmup; i++) {
        func();
    }
    float totalTime = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        func();
        totalTime += timer.getGpuElapsedTimeForPreviousOperation();
    }
    return totalTime / iterations;
}

template <typename Timer, typename Func, typename... Args>
void tuneBlockSize(Timer &timer, Func func, Args &&...args) {
    std::vector<int> blockSizes = {8, 16, 32, 64, 128, 256, 512, 1024};
    float bestTime = std::numeric_limits<float>::max();
    int bestBlockSize = -1;
    for (int blockSize : blockSizes) {
        float time = measureTime(timer, [&]() { func(std::forward<Args>(args)..., blockSize); });
        std::cout << "Block size: " << blockSize << ", time: " << time << " ms" << std::endl;
        if (time < bestTime) {
            bestTime = time;
            bestBlockSize = blockSize;
        }
    }
    std::cout << "Best block size: " << bestBlockSize << std::endl;
}

int main() {
    const int SIZE = 1 << 22;
    const int NPOT = SIZE - 3;
    int *a = new int[SIZE];
    int *b = new int[SIZE];
    int *c = new int[SIZE];

    genArray(SIZE - 1, a, 50);
    a[SIZE - 1] = 0;

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    tuneBlockSize(StreamCompaction::Naive::timer(), StreamCompaction::Naive::scan, SIZE, c, a);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    tuneBlockSize(StreamCompaction::Naive::timer(), StreamCompaction::Naive::scan, NPOT, c, a);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    tuneBlockSize(StreamCompaction::Efficient::timer(), StreamCompaction::Efficient::scan, SIZE, c,
                  a);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    tuneBlockSize(StreamCompaction::Efficient::timer(), StreamCompaction::Efficient::scan, NPOT, c,
                  a);

    genArray(SIZE - 1, a, 4);
    a[SIZE - 1] = 0;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    tuneBlockSize(StreamCompaction::Efficient::timer(), StreamCompaction::Efficient::compact, SIZE,
                  c, a);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    tuneBlockSize(StreamCompaction::Efficient::timer(), StreamCompaction::Efficient::compact, NPOT,
                  c, a);

    system("pause");
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
