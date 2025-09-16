# CUDA Stream Compaction

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

- Yunhao Qian
  - [LinkedIn](https://www.linkedin.com/in/yunhao-qian-026980170/)
  - [GitHub](https://github.com/yunhao-qian)
- Tested on:
  - OS: Windows 11, 24H2
  - CPU: 13th Gen Intel(R) Core(TM) i7-13700 (2.10 GHz)
  - GPU: NVIDIA GeForce RTX 4090
  - RAM: 32.0 GB

## Overview

### Features

This project implements stream compaction and its building blocks (map, scan, scatter) using multiple approaches. Key features include:

- A CPU implementation of scan and stream compaction
- GPU implementations of scan, using both naive and work-efficient methods
- A GPU implementation of stream compaction based on the work-efficient scan
- C++ and Python scripts used to automate performance measurement accurately and programmatically
- Performance analysis comparing the different methods

### Changes to `CMakeLists.txt`

An additional executable,`measure_time.exe`, has been added to the project to support block size tuning, performance benchmarking, and profiling.

### Changes to Function Signatures

To simplify block size tuning, I added an optional `blockSize` parameter to the following functions, each with a tuned default value:

- `Naive::scan(..., const int blockSize = 256)`
- `Efficient::scan(..., const int blockSize = 64)`
- `Efficient::compact(..., const int blockSize = 64)`

These parameters are used only by `measure_time.exe` and do not affect existing calls.

## Part 1: CPU Scan & Stream Compaction

In [`cpu.h`](stream_compaction/cpu.h) and [`cpu.cu`](stream_compaction/cpu.cu):

- `scan()`: Computes an exclusive prefix sum using a simple `for` loop.
- `compactWithoutScan()`: Performs stream compaction directly with a `for` loop, without calling `scan()`.
- `compactWithScan()`: Implements stream compaction using map → scan → scatter. While it follows the structure of a parallel implementation, it is built entirely with `for` loops.

## Part 2:  Naive GPU Scan Algorithm

In [`naive.h`](stream_compaction/naive.h) and [`naive.cu`](stream_compaction/naive.cu):

- `scan()`: Implements the naive algorithm from GPU Gems 3, Section 39.2.1, with the following differences:
  - Uses only global memory (does not leverage shared memory).
  - Launches one kernel per level, plus an additional kernel at the end to shift the results, rather than fusing the entire algorithm into a single kernel.

## Part 3: Work-Efficient GPU Scan & Stream Compaction

### 3.1. Scan

In [`efficient.h`](stream_compaction/efficient.h) and [`efficient.cu`](stream_compaction/efficient.cu):

- `scan()`: Implements the work-efficient algorithm from GPU Gems 3, Section 39.2.2, with the following differences:
  - Uses only global memory (does not leverage shared memory).
  - Launches one kernel per up-sweep/down-sweep level, rather than a single fused kernel.
  - Saves results in place instead of out-of-place.
- Added `scanImpl()`, which operates directly on device arrays. This avoids the CPU buffer interface exposed by `scan()`, making it easier to integrate with CUDA code.

### 3.2. Stream Compaction

In [`common.h`](stream_compaction/common.h) and [`common.cu`](stream_compaction/common.cu):

- `kernMapToBoolean()` A CUDA kernel that maps each integer to 0 or 1, depending on whether the value is zero.
- `kernScatter()`: A CUDA kernel that performs the scatter operation with vector addressing. Conditioned on a boolean array, it optionally stores elements at locations specified by an index array.

In [`efficient.h](stream_compaction/efficient.h) and [`efficient.cu`](stream_compaction/efficient.cu):

- `compact()`: Implements stream compaction on GPU using map (via `kernMapToBoolean()`) → scan (via `scanImpl()`) → scatter (via `kernScatter()`).

## Part 4: Using Thrust's Implementation

In [`thrust.h`](stream_compaction/thrust.h) and [`thrust.cu`](stream_compaction/thrust.cu):

- `scan()`: Wraps `thrust::exclusive_scan()`, adding timing instrumentation and exposing the same API as the other implementations.

## Part 5: Why is My GPU Approach So Slow?

I believe Part 3.1 already incorporates the optimizations described in the instructions:

- For an up-sweep or down-sweep level with a given `offset`, only elements at indices `j = (i + 1) * offset * 2 - 1`, where `i` is any integer and `0 ≤ j < n`, require processing. All other elements can be skipped.
- Instead of mapping each kernel thread to a `j`, we map each thread directly to an `i`. This ensures that nearly all launched threads perform useful work.
- As a result, fewer threads need to be launched. For a given `block_size`, the number of required blocks becomes
`ceil(n / (2 * offset * block_size))`.

## Part 6: Extra Credit

### Extra Credit 1: Radix Sort

This part is not implemented.

### Extra Credit 2: GPU Scan Using Shared Memory && Hardware Optimization

To improve the work-efficient implementation, I developed the work-efficient plus variant in [`efficient_plus.h`](stream_compaction/efficient_plus.h) and [`efficient_plus.cu`](stream_compaction/efficient_plus.cu). Key aspects of this design include:

- Kernel fusing: The up-sweep and down-sweep phases are fused into a single kernel invocation. Although compacting indices no longer reduces the number of blocks, this approach minimizes divergence and is therefore retained.
- Shared memory usage: Instead of operating directly on global memory, data is first copied into shared memory, scanned per block, and then copied back to global memory.
- Recursive tiling: Since a block can only process a limited number of elements, the algorithm is made recursive.
  - The array is partitioned into tiles, one per block.  
  - In addition to performing the scan, each block writes its tile sum into a new array.  
  - An exclusive scan is recursively applied to this array of tile sums, and the results are then added back to the input data.
- Avoiding shared memory bank conflicts: Shared memory indexing is padded following the method described in GPU Gems. This introduces only a small increase in memory usage.
- Preallocating the recursion buffer: To avoid repeated allocation of temporary GPU buffers for tile sums, the recursion depth and total required storage are precomputed. A single contiguous GPU buffer is allocated and reused across all recursion levels.
- Multiple elements per thread: As an experiment, I added support for processing multiple elements per thread, controlled by the compile-time template parameter `ElementsPerThread`. Different variants are dispatched at runtime, and loops over per-thread elements are unrolled with `#pragma unroll`. In practice, this optimization was not beneficial; the tuned configuration still uses one element per thread.

## Part 7: Write-up

Project description: see the [Features](#features) section at the top.

### Performance Analysis

#### `measure_time.exe`

To simplify performance analysis, I added a C++ executable, `measure_time.exe`. The implementation is in [`measure_time.cpp`](src/measure_time.cpp), which:

- Accepts the operation (scan or compact), implementation (CPU, GPU naive, GPU work-efficient, or GPU Thrust), input size, block size, and number of elements per thread as command-line arguments.
- Generates random input data and prints the measured execution time (in milliseconds) to the console.

I created this tool because measuring a configuration only once is often imprecise. In my earlier attempts, running repeated measurements within a C++ loop caused the results to drift significantly. In particular, Thrust measurements became unexpectedly slower, sometimes even slower than the GPU naive implementation. I suspect this was due to frequent GPU memory allocations and deallocations (since the exposed API uses CPU inputs and outputs), which created an atypical workload and put the driver in a degraded performance state.

To avoid this issue, I designed `measure_time.exe` to test only a single configuration with one iteration per program launch. Repeated measurements are instead automated by accompanying Python scripts.

#### Optimizing Parameters

To optimize parameters such as block sizes and the number of elements per thread, I created a Python script, [`tune_parameters.py`](scripts/tune_parameters.py).

- Because optimal block sizes vary with input size, tests are conducted on a fixed scale of $2^{22}$, using both a power-of-two input ($2^{22}$) and a non-power-of-two input ($2^{22} - 3$). The results indicate that the distinction between power-of-two and non-power-of-two input sizes has minimal impact on performance. Therefore, this factor will not be considered further in the discussion.
- Block sizes are sampled over a log-spaced range: 8, 16, …, 512, 1024.
- The number of processed elements per thread is one of 1, 2, 4, 8, and 16.
- Each configuration is executed 10 times, with the mean runtime recorded.
- The final selection balances performance across both power-of-two and non-power-of-two cases.

From these experiments, the chosen defaults are:

- Block size 256 for the naive scan
- Block size 64 for the work-efficient scan and compaction
- Block size 256 for the work-efficient plus (for extra credit 2) scan, with 1 element per thread

#### Performance Comparison

To systematically collect execution time data across many configurations, I created a Python script, [`measure_performance.py`](scripts/measure_performance.py). This script relies on the previously described `measure_time.exe` to benchmark runtime across a range of input sizes and implementations. The procedure is as follows:

- Input sizes are tested from $2^4$ up to $2^{27}$, using both exact powers of two and non-powers of two ($2^i - 3$ for each $i$) to capture different performance behaviors.
- The number of elements processed per thread is selected from the set {1, 2, 4, 8, 16}.
- Each configuration is executed 20 times, and the mean runtime is recorded.
- Results are stored in a JSON file ([`performance.json`](scripts/performance.json)).

The JSON data is then processed by another script, [`plot_performance.py`](scripts/plot_performance.py), which generates the figures shown in this report.

The first figure presents the full dataset:

![performance comparison](img/performance.png)

Because non-power-of-two inputs produce jagged trends, and CPU performance scales on a very different range than GPU performance, a second figure was generated using only GPU data and power-of-two inputs:

![performance comparison GPU-only](img/performance-gpu-only.png)

### Observation and Analysis

#### CPU Implementation

The minimal for-loop CPU implementation consistently demonstrates $O(N)$ complexity across all tested input sizes, as confirmed by the linear trend in the log-log plot. This complexity holds regardless of whether the limiting factor is compute or memory. Regarding the bottleneck:

- The compute workload is very light, since addition operations are fast.
- The memory access pattern is highly favorable, as all reads and writes are sequential.

Although memory is typically slower than arithmetic operations—suggesting the implementation may be slightly memory-bound—the distinction is not critical here.

Importantly, this simple, low-overhead implementation outperforms the GPU variants (naive, work-efficient, and Thrust) for inputs up to about $2^{17}$. The likely reason is the absence of GPU kernel launch overhead, which allows the CPU to handle small and mid-sized inputs more efficiently.

#### Naive & Work-Efficient GPU Implementations

Common characteristics:

- For $N < 2^{20}$, both implementations show limited sensitivity to input size. This is likely because the $O(\log N)$ kernel launches dominate execution time, while the work per kernel remains relatively small. Additionally, the input sizes may be too small to fully utilize GPU resources. In this regime, the bottleneck is neither compute nor memory, but the overhead of repeated kernel launches.
- Beyond $2^{20}$, execution time increases rapidly with $N$, indicating saturation of a GPU resource. Given that both implementations rely heavily on global memory, the performance bottleneck is most likely memory I/O rather than computation.

Comparison of the two:

- For $N < 2^{23}$, the so-called work-efficient implementation is actually slower than the naive version. This can be attributed to kernel launch overhead: the up-sweep and down-sweep phases double the number of kernel invocations. Although the total work is reduced, the benefit is negligible at these smaller sizes.
- For larger $N$, the work-efficient method begins to outperform the naive implementation, and the gap widens quickly. This is because the work-efficient algorithm performs only $O(N)$ total operations, whereas the naive approach requires $O(N \log N)$.

#### Work-Efficient Plus GPU Implementation (for Extra Credit 2)

- For small $N$, the performance is comparable to the naive and work-efficient implementations, without clear advantages. This is because each kernel must perform a full round trip from global memory → shared memory → global memory. When the amount of computation on shared memory is limited, this memory traffic dominates the runtime.  
- For larger inputs ($N \geq 2^{20}$), the benefits of this implementation become evident due to the optimizations described earlier. It even outperforms Thrust at $2^{20} \leq N \leq 2^{25}$. However, I suspect Thrust’s performance in my environments are not representative (see below), so I do not claim that my implementation is more optimized than Thrust in general.

#### Thrust GPU Implementation

To better understand the behavior of `thrust::exclusive_scan()`, I profiled it using Nsight Systems. The results revealed only two CUDA kernel calls:

- `DeviceScanInitKernel`, which is short-lived
- `DeviceScanKernel`, which dominates execution time

This suggests that Thrust fuses the entire scan into just two kernels. While the internal algorithm is not directly visible, it is reasonable to assume that it leverages shared memory and optimized memory access patterns.

The performance profile of Thrust’s `exclusive_scan` is notable:

- Small sizes ($N \leq 2^{17}$): Runtime remains relatively flat across input sizes. Although slower than the CPU implementation, it is still much faster than the naive and work-efficient GPU versions, likely due to the minimal number of kernel launches.
- Mid sizes ($2^{18} \leq N \leq 2^{23}$): Runtime unexpectedly spikes, at times even performing worse than the naive implementation. It seems unlikely that NVIDIA’s official library would be poorly optimized, especially since this anomaly does not appear for other people. I suspect the performance degradation is related to the measurement methodology: repeatedly allocating memory, copying data CPU → GPU, launching kernels, copying results GPU → CPU, and freeing memory. This atypical workflow may interact with the driver in unusual ways. This remains a hypothesis and warrants further investigation.
- Large sizes ($N > 2^{23}$): Thrust significantly outperforms the naive and work-efficient GPU implementations, presumably because its internal use of shared memory and optimized access patterns scales effectively at large input sizes.

#### Outputs of `cis5650_stream_compaction_test.exe`

```text
****************
** SCAN TESTS **
****************
    [   2  43  36  34   6  47  28  21  20  47   9  14  36 ...  28   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.6333ms    (std::chrono Measured)
    [   0   2  45  81 115 121 168 196 217 237 284 293 307 ... 102726393 102726421 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.4916ms    (std::chrono Measured)
    [   0   2  45  81 115 121 168 196 217 237 284 293 307 ... 102726316 102726346 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.335968ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.330656ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.464192ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.368544ms    (CUDA Measured)
    passed
==== work-efficient plus scan, power-of-two ====
   elapsed time: 0.191456ms    (CUDA Measured)
    passed
==== work-efficient plus scan, non-power-of-two ====
   elapsed time: 0.073984ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.320512ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.336736ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   2   2   2   1   2   3   0   3   1   2   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 7.1751ms    (std::chrono Measured)
    [   3   2   2   2   1   2   3   3   1   2   2   2   3 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 7.0687ms    (std::chrono Measured)
    [   3   2   2   2   1   2   3   3   1   2   2   2   3 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 14.5221ms    (std::chrono Measured)
    [   3   2   2   2   1   2   3   3   1   2   2   2   3 ...   1   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.524ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.521024ms    (CUDA Measured)
    passed
Press any key to continue . . .
```

#### Outputs of `tune_parameters.py`

```text
Naive scan, power-of-two
Block size: 8, time: 6.210713481 ms
Block size: 16, time: 3.168515182 ms
Block size: 32, time: 1.640864002 ms
Block size: 64, time: 0.8767040014999999 ms
Block size: 128, time: 0.5051392019 ms
Block size: 256, time: 0.3497567982 ms
Block size: 512, time: 0.3369727998 ms
Block size: 1024, time: 0.3949791968 ms
Optimal block size: 512, time: 0.3369727998 ms
========================================
Naive scan, non-power-of-two
Block size: 8, time: 6.208518411000001 ms
Block size: 16, time: 3.1698239809999995 ms
Block size: 32, time: 1.650400006 ms
Block size: 64, time: 0.8862239956 ms
Block size: 128, time: 0.5067872018 ms
Block size: 256, time: 0.3504383981 ms
Block size: 512, time: 0.346905601 ms
Block size: 1024, time: 0.4155999988 ms
Optimal block size: 512, time: 0.346905601 ms
========================================
Work-efficient scan, power-of-two
Block size: 8, time: 0.9049055992999999 ms
Block size: 16, time: 0.7210080087999999 ms
Block size: 32, time: 0.6597408055 ms
Block size: 64, time: 0.5313376009999999 ms
Block size: 128, time: 0.6298431963000001 ms
Block size: 256, time: 0.6812607945 ms
Block size: 512, time: 0.5833792001 ms
Block size: 1024, time: 0.6668896109 ms
Optimal block size: 64, time: 0.5313376009999999 ms
========================================
Work-efficient scan, non-power-of-two
Block size: 8, time: 0.9357727944 ms
Block size: 16, time: 0.7128639936000001 ms
Block size: 32, time: 0.7185056090999999 ms
Block size: 64, time: 0.5883967997000001 ms
Block size: 128, time: 0.5995808003 ms
Block size: 256, time: 0.6008256017000001 ms
Block size: 512, time: 0.6760576010000001 ms
Block size: 1024, time: 0.5939391851 ms
Optimal block size: 64, time: 0.5883967997000001 ms
========================================
Work-efficient compact, power-of-two
Block size: 8, time: 1.417711996 ms
Block size: 16, time: 0.9495487984000001 ms
Block size: 32, time: 0.7181632041999999 ms
Block size: 64, time: 0.6332575947 ms
Block size: 128, time: 0.6355327934999999 ms
Block size: 256, time: 0.577478391 ms
Block size: 512, time: 0.7434591980000002 ms
Block size: 1024, time: 0.6726783932 ms
Optimal block size: 256, time: 0.577478391 ms
========================================
Work-efficient compact, non-power-of-two
Block size: 8, time: 1.423475218 ms
Block size: 16, time: 0.9857856036999998 ms
Block size: 32, time: 0.8006624043 ms
Block size: 64, time: 0.5751327993 ms
Block size: 128, time: 0.6924479992 ms
Block size: 256, time: 0.5636575996 ms
Block size: 512, time: 0.6784191968000001 ms
Block size: 1024, time: 0.6932192027999999 ms
Optimal block size: 256, time: 0.5636575996 ms
========================================
Work-efficient plus scan, power-of-two
Block size: 8, elements per thread: 1, time: 0.5458272011 ms
Block size: 8, elements per thread: 2, time: 0.3792447983 ms
Block size: 8, elements per thread: 4, time: 0.3159423963 ms
Block size: 8, elements per thread: 8, time: 0.3622208059 ms
Block size: 8, elements per thread: 16, time: 0.4684351951 ms
Block size: 16, elements per thread: 1, time: 0.2957440004 ms
Block size: 16, elements per thread: 2, time: 0.2765696033999999 ms
Block size: 16, elements per thread: 4, time: 0.3591648042 ms
Block size: 16, elements per thread: 8, time: 0.3541247966 ms
Block size: 16, elements per thread: 16, time: 0.4408064038 ms
Block size: 32, elements per thread: 1, time: 0.2547711969 ms
Block size: 32, elements per thread: 2, time: 0.2487744019 ms
Block size: 32, elements per thread: 4, time: 0.2829280005 ms
Block size: 32, elements per thread: 8, time: 0.355180803 ms
Block size: 32, elements per thread: 16, time: 0.4990495979999999 ms
Block size: 64, elements per thread: 1, time: 0.21321919859999997 ms
Block size: 64, elements per thread: 2, time: 0.23418560179999998 ms
Block size: 64, elements per thread: 4, time: 0.26373759820000003 ms
Block size: 64, elements per thread: 8, time: 0.3173408001 ms
Block size: 64, elements per thread: 16, time: 0.45957120059999995 ms
Block size: 128, elements per thread: 1, time: 0.2471552 ms
Block size: 128, elements per thread: 2, time: 0.21002880040000002 ms
Block size: 128, elements per thread: 4, time: 0.2764960006 ms
Block size: 128, elements per thread: 8, time: 0.3514591991999999 ms
Block size: 128, elements per thread: 16, time: 0.49059520079999996 ms
Block size: 256, elements per thread: 1, time: 0.19113280039999997 ms
Block size: 256, elements per thread: 2, time: 0.2154880017 ms
Block size: 256, elements per thread: 4, time: 0.2787679984 ms
Block size: 256, elements per thread: 8, time: 0.3731999963 ms
Block size: 256, elements per thread: 16, time: 0.49939519470000004 ms
Block size: 512, elements per thread: 1, time: 0.20608639850000002 ms
Block size: 512, elements per thread: 2, time: 0.2563392014 ms
Block size: 512, elements per thread: 4, time: 0.3583200006 ms
Block size: 512, elements per thread: 8, time: 0.3594367981 ms
Block size: 512, elements per thread: 16 - crashed
Block size: 1024, elements per thread: 1, time: 0.234352003 ms
Block size: 1024, elements per thread: 2, time: 0.25858239680000006 ms
Block size: 1024, elements per thread: 4, time: 0.3363999992000001 ms
Block size: 1024, elements per thread: 8 - crashed
Block size: 1024, elements per thread: 16 - crashed
Optimal config: block size 256, elements per thread 1, time: 0.19113280039999997 ms
========================================
Work-efficient plus scan, non-power-of-two
Block size: 8, elements per thread: 1, time: 0.4914176017 ms
Block size: 8, elements per thread: 2, time: 0.32388479719999996 ms
Block size: 8, elements per thread: 4, time: 0.38091839839999997 ms
Block size: 8, elements per thread: 8, time: 0.33217279899999996 ms
Block size: 8, elements per thread: 16, time: 0.39384639860000004 ms
Block size: 16, elements per thread: 1, time: 0.3107008011 ms
Block size: 16, elements per thread: 2, time: 0.2454912007 ms
Block size: 16, elements per thread: 4, time: 0.3092735976 ms
Block size: 16, elements per thread: 8, time: 0.32160959829999997 ms
Block size: 16, elements per thread: 16, time: 0.42109759750000003 ms
Block size: 32, elements per thread: 1, time: 0.26428479559999996 ms
Block size: 32, elements per thread: 2, time: 0.25905919679999995 ms
Block size: 32, elements per thread: 4, time: 0.3214975982 ms
Block size: 32, elements per thread: 8, time: 0.3603167982 ms
Block size: 32, elements per thread: 16, time: 0.4624383956 ms
Block size: 64, elements per thread: 1, time: 0.20346560030000002 ms
Block size: 64, elements per thread: 2, time: 0.2914719998 ms
Block size: 64, elements per thread: 4, time: 0.29802560209999995 ms
Block size: 64, elements per thread: 8, time: 0.3288128018 ms
Block size: 64, elements per thread: 16, time: 0.449561593 ms
Block size: 128, elements per thread: 1, time: 0.20832960009999998 ms
Block size: 128, elements per thread: 2, time: 0.24085119969999996 ms
Block size: 128, elements per thread: 4, time: 0.2845311985 ms
Block size: 128, elements per thread: 8, time: 0.3271040023 ms
Block size: 128, elements per thread: 16, time: 0.46438719919999993 ms
Block size: 256, elements per thread: 1, time: 0.1988800004 ms
Block size: 256, elements per thread: 2, time: 0.24773760570000003 ms
Block size: 256, elements per thread: 4, time: 0.3005983978 ms
Block size: 256, elements per thread: 8, time: 0.3462976009 ms
Block size: 256, elements per thread: 16, time: 0.48608960519999994 ms
Block size: 512, elements per thread: 1, time: 0.2411488013 ms
Block size: 512, elements per thread: 2, time: 0.2356735976 ms
Block size: 512, elements per thread: 4, time: 0.31453439899999996 ms
Block size: 512, elements per thread: 8, time: 0.36263679849999997 ms
Block size: 512, elements per thread: 16 - crashed
Block size: 1024, elements per thread: 1, time: 0.2502848000000001 ms
Block size: 1024, elements per thread: 2, time: 0.25420480219999997 ms
Block size: 1024, elements per thread: 4, time: 0.3396863997 ms
Block size: 1024, elements per thread: 8 - crashed
Block size: 1024, elements per thread: 16 - crashed
Optimal config: block size 256, elements per thread 1, time: 0.1988800004 ms
```
