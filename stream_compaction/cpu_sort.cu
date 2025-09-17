#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "cpu_sort.h"

namespace StreamCompaction {
    namespace CPUSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        void sort(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            std::copy_n(idata, n, odata);
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }
    }
}
