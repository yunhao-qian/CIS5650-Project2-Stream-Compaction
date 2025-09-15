import json
from pathlib import Path

from helper import measure_time


def main() -> None:
    performance_json_path = Path(__file__).parent / "performance.json"
    if performance_json_path.exists():
        with performance_json_path.open(encoding="utf-8") as f:
            performance_data = json.load(f)
    else:
        performance_data = []

    def find_existing_entry(
        operation: str, implementation: str, input_size: int
    ) -> dict | None:
        for entry in performance_data:
            if (
                entry["operation"] == operation
                and entry["implementation"] == implementation
                and entry["input_size"] == input_size
            ):
                return entry
        return None

    for implementation in ["cpu", "naive", "efficient", "thrust"]:
        for exponent in range(4, 28):
            base_input_size = 1 << exponent
            for input_size in [base_input_size, base_input_size - 3]:
                config = {
                    "operation": "scan",
                    "implementation": implementation,
                    "input_size": input_size,
                }
                if find_existing_entry(**config) is not None:
                    print(f"Skipping existing entry: {config}")
                    continue

                time_sum = 0.0
                time_count = 50
                for _ in range(time_count):
                    time_sum += measure_time("scan", implementation, input_size)
                time = time_sum / time_count
                print(f"Measured: {config}, time: {time} ms")
                performance_data.append({**config, "time": time})

                with performance_json_path.open("w", encoding="utf-8") as f:
                    json.dump(performance_data, f, indent=4)


if __name__ == "__main__":
    main()
