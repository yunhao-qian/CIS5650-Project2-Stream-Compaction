#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScan(int n, int offset, int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int beforeIndex = index - offset;
            odata[index] = idata[index] + (beforeIndex < 0 ? 0 : idata[beforeIndex]);
        }

        __global__ void kernShiftRight(int n, int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = index == 0 ? 0 : idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, const int blockSize) {
            const auto dataSize = n * sizeof(int);

            int *dev_iData;
            cudaMalloc((void **)&dev_iData, dataSize);
            cudaMemcpy(dev_iData, idata, dataSize, cudaMemcpyHostToDevice);
            int *dev_oData;
            cudaMalloc((void **)&dev_oData, dataSize);

            timer().startGpuTimer();
            const int gridSize = (n + blockSize - 1) / blockSize;
            for (int offset = 1; offset < n; offset *= 2) {
                kernScan<<<gridSize, blockSize>>>(n, offset, dev_oData, dev_iData);
                std::swap(dev_oData, dev_iData);
            }
            kernShiftRight<<<gridSize, blockSize>>>(n, dev_oData, dev_iData);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_oData, dataSize, cudaMemcpyDeviceToHost);
            cudaFree(dev_iData);
            cudaFree(dev_oData);
        }
    }
}
