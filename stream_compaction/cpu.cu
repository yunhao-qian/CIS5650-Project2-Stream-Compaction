#include <cstdio>

#include "common.h"
#include "cpu.h"

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
            int oCount = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[oCount] = idata[i];
                    ++oCount;
                }
            }
            timer().endCpuTimer();
            return oCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *bools = new int[n];
            int *indices = new int[n];
            timer().startCpuTimer();
            // Step 1: map
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }
            // Step 2: scan
            scan(n, indices, bools);
            // Step 3: scatter
            int oCount = 0;
            for (int i = 0; i < n; ++i) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                    ++oCount;
                }
            }
            timer().endCpuTimer();
            delete[] bools;
            delete[] indices;
            return oCount;
        }
    }
}
