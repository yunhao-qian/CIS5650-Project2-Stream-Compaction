#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient_plus.h"
#include "radix_sort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernExtractBit(int n, int mask, int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = (idata[index] & mask) == 0 ? 0 : 1;
        }

        __global__ void kernScatter(int n, unsigned mask, int zeroCount, const int *positions,
                                    const int *idata, int *odata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            bool isOne = (static_cast<unsigned>(idata[index]) & mask) != 0;
            if (isOne) {
                odata[zeroCount + positions[index]] = idata[index];
            } else {
                int zeros_before = index - positions[index];
                odata[zeros_before] = idata[index];
            }
        }

        void sort(int n, int *odata, const int *idata) {
            const auto dataSize = n * sizeof(int);

            int *dev_iData;
            cudaMalloc((void **)&dev_iData, dataSize);
            cudaMemcpy(dev_iData, idata, dataSize, cudaMemcpyHostToDevice);

            int *dev_oData;
            cudaMalloc((void **)&dev_oData, dataSize);

            int *dev_positions;
            cudaMalloc((void **)&dev_positions, dataSize);

            timer().startGpuTimer();
            const int blockSize = 256;
            const int gridSize = (n + blockSize - 1) / blockSize;
            for (int bit = 0; bit < 32; ++bit) {
                const unsigned mask = 1u << bit;
                kernExtractBit<<<gridSize, blockSize>>>(n, mask, dev_positions, dev_iData);
                int lastBit;
                cudaMemcpy(&lastBit, &dev_positions[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
                EfficientPlus::scanDeviceInPlace(n, dev_positions);
                int oneCount;
                cudaMemcpy(&oneCount, &dev_positions[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
                oneCount += lastBit;
                int zeroCount = n - oneCount;
                kernScatter<<<gridSize, blockSize>>>(n, mask, zeroCount, dev_positions, dev_iData,
                                                     dev_oData);
                std::swap(dev_iData, dev_oData);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_oData, dataSize, cudaMemcpyDeviceToHost);
            cudaFree(dev_iData);
            cudaFree(dev_oData);
            cudaFree(dev_positions);
        }
    }
}
