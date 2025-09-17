#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory_resource.h>

#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Use a memory pool to reduce memory allocation overhead of Thrust operations.
            static thrust::cuda::memory_resource upstream;
            static thrust::mr::new_delete_resource bookkeeper;
            static thrust::mr::disjoint_unsynchronized_pool_resource pool(&upstream, &bookkeeper);
            thrust::mr::allocator<char, decltype(pool)> allocator(&pool);
            const auto policy = thrust::cuda::par(allocator);

            thrust::device_vector<int> dev_data(n);
            thrust::copy_n(idata, n, dev_data.begin());
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            thrust::exclusive_scan(policy, dev_data.begin(), dev_data.end(), dev_data.begin());
            timer().endGpuTimer();

            thrust::copy(dev_data.begin(), dev_data.end(), odata);
        }
    }
}
