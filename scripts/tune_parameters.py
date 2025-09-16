import subprocess

from helper import measure_time


def tune_block_size(operation: str, implementation: str, input_size: int) -> None:
    block_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    best_time = float("inf")
    best_block_size = -1
    for block_size in block_sizes:
        time_sum = 0.0
        time_count = 10
        for _ in range(time_count):
            time_sum += measure_time(operation, implementation, input_size, block_size)
        time = time_sum / time_count
        print(f"Block size: {block_size}, time: {time} ms")
        if time < best_time:
            best_time = time
            best_block_size = block_size
    print(f"Optimal block size: {best_block_size}, time: {best_time} ms")


def tune_efficient_plus(input_size: int) -> None:
    block_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    elements_per_thread_values = [1, 2, 4, 8, 16]
    best_time = float("inf")
    best_config = (-1, -1)
    for block_size in block_sizes:
        for elements_per_thread in elements_per_thread_values:
            time_sum = 0.0
            time_count = 10
            try:
                for _ in range(time_count):
                    time_sum += measure_time(
                        "scan",
                        "efficient_plus",
                        input_size,
                        block_size,
                        elements_per_thread,
                    )
            except subprocess.CalledProcessError:
                print(
                    f"Block size: {block_size}, "
                    f"elements per thread: {elements_per_thread} "
                    "- crashed"
                )
                continue
            time = time_sum / time_count
            print(
                f"Block size: {block_size}, "
                f"elements per thread: {elements_per_thread}, "
                f"time: {time} ms"
            )
            if time < best_time:
                best_time = time
                best_config = (block_size, elements_per_thread)
    print(
        f"Optimal config: block size {best_config[0]}, "
        f"elements per thread {best_config[1]}, "
        f"time: {best_time} ms"
    )


def print_divider() -> None:
    print("=" * 40)


def main():
    input_size = 1 << 22

    print("Naive scan, power-of-two")
    tune_block_size("scan", "naive", input_size)
    print_divider()
    print("Naive scan, non-power-of-two")
    tune_block_size("scan", "naive", input_size - 3)
    print_divider()
    print("Work-efficient scan, power-of-two")
    tune_block_size("scan", "efficient", input_size)
    print_divider()
    print("Work-efficient scan, non-power-of-two")
    tune_block_size("scan", "efficient", input_size - 3)
    print_divider()
    print("Work-efficient compact, power-of-two")
    tune_block_size("compact", "efficient", input_size)
    print_divider()
    print("Work-efficient compact, non-power-of-two")
    tune_block_size("compact", "efficient", input_size - 3)
    print_divider()
    print("Work-efficient plus scan, power-of-two")
    tune_efficient_plus(input_size)
    print_divider()
    print("Work-efficient plus scan, non-power-of-two")
    tune_efficient_plus(input_size - 3)


if __name__ == "__main__":
    main()
