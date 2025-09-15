import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


def main():
    json_path = Path(__file__).parent / "performance_comparison.nljson"
    with json_path.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    def plot_and_save(
        methods: list[str], labels: list[str], power_of_two_only: bool, filename: str
    ):
        fig, ax = plt.subplots()
        sample_counts = []
        for method, label in zip(methods, labels):
            n_values = []
            time_values = []
            for entry in data:
                if entry["method"] == method:
                    if not power_of_two_only or (entry["n"] & (entry["n"] - 1)) == 0:
                        n_values.append(entry["n"])
                        time_values.append(entry["time"])

            n_values_sorted = sorted(set(n_values))
            time_values_sorted = []
            for n in n_values_sorted:
                time_samples = []
                for n_value, time_value in zip(n_values, time_values):
                    if n_value == n:
                        time_samples.append(time_value)
                sample_counts.append(len(time_samples))
                time_values_sorted.append(np.mean(time_samples))
            ax.plot(
                n_values_sorted,
                time_values_sorted,
                label=label,
                marker="o",
                markersize=4,
            )

        print("Average sample count:", np.mean(sample_counts))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, which="both", ls="--", lw=0.5)
        ax.set_xlabel("Input size")
        ax.set_ylabel("Time (ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(__file__).parent.parent / "img" / filename)

    all_methods = ["cpu", "naive", "efficient", "thrust"]
    all_labels = ["CPU", "GPU (naive)", "GPU (work-efficient)", "GPU (Thrust)"]
    plot_and_save(all_methods, all_labels, False, "performance-comparison.png")
    plot_and_save(
        all_methods[1:], all_labels[1:], True, "performance-comparison-gpu-only.png"
    )


if __name__ == "__main__":
    main()
