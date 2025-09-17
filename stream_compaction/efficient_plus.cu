#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient_plus.h"

namespace StreamCompaction {
    namespace EfficientPlus {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __host__ __device__ unsigned conflictFreeIndex(unsigned index) {
            // Number of banks: 32
            unsigned offset = index / 32U;
            offset += offset / 32U;
            return index + offset;
        }

        __device__ int &conflictFreeGet(int *data, unsigned index) {
            return data[conflictFreeIndex(index)];
        }

        template <unsigned ElementsPerThread>
        __global__ void scanPerBlock(unsigned n, int *__restrict__ data, int *__restrict__ sums) {
            unsigned tileSize = blockDim.x * ElementsPerThread * 2U;

            extern __shared__ int sharedData[];
            unsigned startSharedIndex = threadIdx.x * ElementsPerThread * 2U;
            unsigned endSharedIndex = startSharedIndex + ElementsPerThread * 2U;
            unsigned startGlobalIndex = blockIdx.x * tileSize + startSharedIndex;
            unsigned endGlobalIndex = min(startGlobalIndex + ElementsPerThread * 2U, n);
            {
                // Copy data to shared memory.
                unsigned sharedIndex = startSharedIndex;
                unsigned globalIndex = startGlobalIndex;
                for (; globalIndex < endGlobalIndex; ++sharedIndex, ++globalIndex) {
                    conflictFreeGet(sharedData, sharedIndex) = data[globalIndex];
                }
                for (; sharedIndex < endSharedIndex; ++sharedIndex) {
                    conflictFreeGet(sharedData, sharedIndex) = 0;
                }
            }
            __syncthreads();

            // Up-sweep
            for (unsigned offset = 1U; offset < tileSize; offset *= 2U) {
#pragma unroll
                for (unsigned i = 0U; i < ElementsPerThread; ++i) {
                    unsigned index = (threadIdx.x * ElementsPerThread + i + 1U) * offset * 2U - 1U;
                    if (index < tileSize) {
                        conflictFreeGet(sharedData, index) +=
                            conflictFreeGet(sharedData, index - offset);
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0U) {
                int &lastElement = conflictFreeGet(sharedData, tileSize - 1U);
                // Save the total sum of this block to the sums array.
                if (sums != nullptr) {
                    // sums may be nullptr for the last recursion.
                    sums[blockIdx.x] = lastElement;
                }
                // Clear the last element before down-sweep.
                lastElement = 0;
            }
            __syncthreads();

            // Down-sweep
            for (unsigned offset = tileSize / 2U; offset > 0U; offset /= 2U) {
#pragma unroll
                for (unsigned i = 0U; i < ElementsPerThread; ++i) {
                    unsigned index = (threadIdx.x * ElementsPerThread + i + 1U) * offset * 2U - 1U;
                    if (index < tileSize) {
                        int &leftChild = conflictFreeGet(sharedData, index - offset);
                        int &rightChild = conflictFreeGet(sharedData, index);
                        int oldLeftChild = leftChild;
                        leftChild = rightChild;
                        rightChild += oldLeftChild;
                    }
                }
                __syncthreads();
            }

            // Write results back to global memory.
            {
                unsigned sharedIndex = startSharedIndex;
                unsigned globalIndex = startGlobalIndex;
                for (; globalIndex < endGlobalIndex; ++sharedIndex, ++globalIndex) {
                    data[globalIndex] = conflictFreeGet(sharedData, sharedIndex);
                }
            }
        }

        template <unsigned ElementsPerThread>
        __global__ void addSums(unsigned n, int *__restrict__ data, const int *__restrict__ sums) {
            unsigned startIndex = (blockIdx.x * blockDim.x + threadIdx.x) * ElementsPerThread * 2U;
            unsigned endIndex = min(startIndex + ElementsPerThread * 2U, n);
            int sum = sums[blockIdx.x];
            for (unsigned index = startIndex; index < endIndex; ++index) {
                data[index] += sum;
            }
        }

        template <unsigned ElementsPerThread>
        void scanImpl(unsigned n, int *data, unsigned blockSize, unsigned tileSize,
                      unsigned gridSize, int *sums) {
            unsigned actualBlockSize = blockSize;
            if (gridSize == 1U) {
                while (true) {
                    unsigned nextBlockSize = actualBlockSize / 2U;
                    if (nextBlockSize * ElementsPerThread * 2U < n) {
                        break;
                    }
                    actualBlockSize = nextBlockSize;
                }
            }
            unsigned sharedMemorySize =
                (conflictFreeIndex(actualBlockSize * ElementsPerThread * 2U - 1U) + 1U) *
                sizeof(int);
            scanPerBlock<ElementsPerThread>
                <<<gridSize, actualBlockSize, sharedMemorySize>>>(n, data, sums);
            checkCUDAErrorFn("scanPerBlock kernel failed!");
            if (gridSize > 1U) {
                unsigned nextGridSize = (gridSize + tileSize - 1U) / tileSize;
                int *nextSums = nullptr;
                if (nextGridSize > 1U) {
                    nextSums = sums + gridSize;
                }
                scanImpl<ElementsPerThread>(gridSize, sums, blockSize, tileSize, nextGridSize,
                                            nextSums);
                addSums<ElementsPerThread><<<gridSize, blockSize>>>(n, data, sums);
                checkCUDAErrorFn("addSums kernel failed!");
            }
        }

        void scan(int n, int *odata, const int *idata, int blockSize, int elementsPerThread) {
            // Set default parameters.
            if (blockSize <= 0) {
                blockSize = 256;
            }
            if (elementsPerThread <= 0) {
                elementsPerThread = 1;
            }

            const auto dataSize = n * sizeof(int);

            int *dev_data;
            cudaMalloc((void **)&dev_data, dataSize);
            checkCUDAErrorFn("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, dataSize, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy to device failed!");

            // Allocate GPU memory for sums beforehand.
            unsigned tileSize = blockSize * elementsPerThread * 2U;
            unsigned gridSize = (n + tileSize - 1U) / tileSize;

            unsigned totalSumCount = 0U;
            {
                unsigned gridSize = n;
                while (true) {
                    gridSize = (gridSize + tileSize - 1U) / tileSize;
                    if (gridSize <= 1U) {
                        break;
                    }
                    totalSumCount += gridSize;
                }
            }
            int *dev_sums = nullptr;
            if (totalSumCount > 0U) {
                cudaMalloc((void **)&dev_sums, totalSumCount * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_sums failed!");
            }
            cudaDeviceSynchronize();

#define DISPATCH(N)                                                                                \
    case N:                                                                                        \
        scanImpl<N>(n, dev_data, blockSize, tileSize, gridSize, dev_sums);                         \
        break

            timer().startGpuTimer();

            switch (elementsPerThread) {
                DISPATCH(1);
                DISPATCH(2);
                DISPATCH(4);
                DISPATCH(8);
                DISPATCH(16);
            default:
                printf("Unsupported elementsPerThread: %d\n", elementsPerThread);
                exit(1);
            }
            timer().endGpuTimer();

#undef DISPATCH

            cudaMemcpy(odata, dev_data, dataSize, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy to host failed!");
            cudaFree(dev_data);
            checkCUDAErrorFn("cudaFree failed!");
            if (dev_sums != nullptr) {
                cudaFree(dev_sums);
                checkCUDAErrorFn("cudaFree failed!");
            }
        }

        void scanDeviceInPlace(int n, int *data) {
            const unsigned blockSize = 256;
            constexpr unsigned ElementsPerThread = 1;

            unsigned tileSize = blockSize * ElementsPerThread * 2U;
            unsigned gridSize = (n + tileSize - 1U) / tileSize;

            unsigned totalSumCount = 0U;
            {
                unsigned gridSize = n;
                while (true) {
                    gridSize = (gridSize + tileSize - 1U) / tileSize;
                    if (gridSize <= 1U) {
                        break;
                    }
                    totalSumCount += gridSize;
                }
            }
            int *dev_sums = nullptr;
            if (totalSumCount > 0U) {
                cudaMalloc((void **)&dev_sums, totalSumCount * sizeof(int));
                checkCUDAErrorFn("cudaMalloc dev_sums failed!");
            }

            scanImpl<ElementsPerThread>(n, data, blockSize, tileSize, gridSize, dev_sums);

            if (dev_sums != nullptr) {
                cudaFree(dev_sums);
                checkCUDAErrorFn("cudaFree failed!");
            }
        }
    }
}
