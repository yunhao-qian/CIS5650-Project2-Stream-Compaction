import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def is_power_of_two(n: int) -> bool:
    return (n & (n - 1)) == 0 and n != 0


def main() -> None:
    json_path = Path(__file__).parent / "performance.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    def plot_and_save(
        implementations: list[str],
        labels: list[str],
        power_of_two_only: bool,
        filename: str,
    ):
        fig, ax = plt.subplots()
        for implementation, label in zip(implementations, labels):
            n_values = []
            time_values = []
            for entry in data:
                if entry["implementation"] == implementation:
                    if not power_of_two_only or is_power_of_two(entry["input_size"]):
                        n_values.append(entry["input_size"])
                        time_values.append(entry["time"])

            n_values = np.array(n_values)
            time_values = np.array(time_values)
            sort_indices = np.argsort(n_values)
            n_values = n_values[sort_indices]
            time_values = time_values[sort_indices]
            ax.plot(n_values, time_values, label=label, marker="o", markersize=4)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, which="both", ls="--", lw=0.5)
        ax.set_xlabel("Input size")
        ax.set_ylabel("Time (ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(__file__).parent.parent / "img" / filename)

    plot_and_save(
        ["naive", "efficient", "efficient_plus", "thrust", "cpu"],
        [
            "GPU (naive)",
            "GPU (efficient)",
            "GPU (work-efficient plus)",
            "GPU (Thrust)",
            "CPU",
        ],
        False,
        "performance.png",
    )
    plot_and_save(
        ["naive", "efficient", "efficient_plus", "thrust"],
        ["Naive", "Work-efficient", "Work-efficient plus", "Thrust"],
        True,
        "performance-gpu-only.png",
    )


if __name__ == "__main__":
    main()
