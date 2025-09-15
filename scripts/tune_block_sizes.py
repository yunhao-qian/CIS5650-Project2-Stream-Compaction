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


if __name__ == "__main__":
    main()
