#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer &timer();

        void scan(int n, int *odata, const int *idata, int blockSize);

        inline void scan(int n, int *odata, const int *idata) { scan(n, odata, idata, -1); }

        int compact(int n, int *odata, const int *idata, int blockSize);

        inline int compact(int n, int *odata, const int *idata) {
            return compact(n, odata, idata, -1);
        }
    }
}
