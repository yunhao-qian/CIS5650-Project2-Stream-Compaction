#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int *data) {
            // Avoid integer overflows.
            auto indexLL = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            indexLL = (indexLL + 1) * offset * 2 - 1;
            if (indexLL >= n) {
                return;
            }
            int index = static_cast<int>(indexLL);
            int beforeIndex = index - offset;
            if (beforeIndex >= 0) {
                data[index] += data[beforeIndex];
            }
        }

        __global__ void kernDownSweep(int n, int offset, int *data) {
            // Avoid integer overflows.
            auto indexLL = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
            indexLL = (indexLL + 1) * offset * 2 - 1;
            if (indexLL >= n) {
                return;
            }
            int index = static_cast<int>(indexLL);
            int beforeIndex = index - offset;
            if (beforeIndex >= 0) {
                int beforeValue = data[beforeIndex];
                data[beforeIndex] = data[index];
                data[index] += beforeValue;
            }
        }

        void scanImpl(int n, int *data, const int blockSize) {
            const auto computeGridSize = [=](int offset) {
                // Avoid integer overflows when n and blockSize are large.
                const auto divisor = 2LL * offset * blockSize;
                return static_cast<int>((n + divisor - 1) / divisor);
            };

            // Up-sweep
            for (int offset = 1; offset < n; offset *= 2) {
                const int gridSize = computeGridSize(offset);
                kernUpSweep<<<gridSize, blockSize>>>(n, offset, data);
            }
            // Set the last element to 0.
            cudaMemset(data + (n - 1), 0, sizeof(int));
            // Down-sweep
            for (int offset = n / 2; offset >= 1; offset /= 2) {
                const int gridSize = computeGridSize(offset);
                kernDownSweep<<<gridSize, blockSize>>>(n, offset, data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, const int blockSize) {
            const auto dataSize = n * sizeof(int);
            const int ceiledN = 1 << ilog2ceil(n);
            const auto ceiledDataSize = ceiledN * sizeof(int);

            int *dev_data;
            cudaMalloc((void **)&dev_data, ceiledDataSize);
            cudaMemcpy(dev_data, idata, dataSize, cudaMemcpyHostToDevice);
            if (ceiledDataSize > dataSize) {
                cudaMemset(dev_data + n, 0, ceiledDataSize - dataSize);
            }

            timer().startGpuTimer();
            scanImpl(ceiledN, dev_data, blockSize);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, dataSize, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata, const int blockSize) {
            const auto dataSize = n * sizeof(int);
            const int ceiledN = 1 << ilog2ceil(n);
            const auto ceiledDataSize = ceiledN * sizeof(int);

            const int gridSize = (n + blockSize - 1) / blockSize;

            int *dev_iData;
            cudaMalloc((void **)&dev_iData, dataSize);
            cudaMemcpy(dev_iData, idata, dataSize, cudaMemcpyHostToDevice);
            int *dev_bools;
            cudaMalloc((void **)&dev_bools, dataSize);
            int *dev_indices;
            cudaMalloc((void **)&dev_indices, ceiledDataSize);
            int *dev_oData;
            cudaMalloc((void **)&dev_oData, dataSize);

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_iData);
            cudaMemcpy(dev_indices, dev_bools, dataSize, cudaMemcpyDeviceToDevice);
            if (ceiledDataSize > dataSize) {
                cudaMemset(dev_indices + n, 0, ceiledDataSize - dataSize);
            }
            scanImpl(ceiledN, dev_indices, blockSize);
            Common::kernScatter<<<gridSize, blockSize>>>(n, dev_oData, dev_iData, dev_bools,
                                                         dev_indices);
            timer().endGpuTimer();

            int oCount;
            {
                int lastBool;
                cudaMemcpy(&lastBool, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int lastIndex;
                cudaMemcpy(&lastIndex, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                oCount = lastIndex + lastBool;
            }
            if (oCount > 0) {
                cudaMemcpy(odata, dev_oData, oCount * sizeof(int), cudaMemcpyDeviceToHost);
            }
            cudaFree(dev_iData);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_oData);
            return oCount;
        }
    }
}
