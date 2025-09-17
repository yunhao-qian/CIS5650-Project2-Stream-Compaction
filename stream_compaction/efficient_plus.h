#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientPlus {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, int blockSize, int elementsPerThread);

        inline void scan(int n, int *odata, const int *idata) { scan(n, odata, idata, -1, -1); }

        void scanDeviceInPlace(int n, int *data);
    }
}
