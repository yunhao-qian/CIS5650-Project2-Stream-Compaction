#include "cpu.h"
#include <cstdio>

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer() {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your
         * GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int o_count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[o_count] = idata[i];
                    ++o_count;
                }
            }
            timer().endCpuTimer();
            return o_count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // Step 1: map
            int *bools = new int[n];
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }
            // Step 2: scan
            int *indices = new int[n];
            scan(n, indices, bools);
            // Step 3: scatter
            int o_count = 0;
            for (int i = 0; i < n; ++i) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                    ++o_count;
                }
            }
            timer().endCpuTimer();
            return o_count;
        }
    }
}
