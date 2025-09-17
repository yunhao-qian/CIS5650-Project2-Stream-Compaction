#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, int blockSize);

        inline void scan(int n, int *odata, const int *idata) { scan(n, odata, idata, -1); }
    }
}
