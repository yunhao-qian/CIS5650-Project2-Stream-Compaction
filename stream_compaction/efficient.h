#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, const int blockSize = 128);

        int compact(int n, int *odata, const int *idata, const int blockSize = 512);
    }
}
