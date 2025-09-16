#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientPlus {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, const int blockSize = 256,
                  const int elementsPerThread = 1);
    }
}
