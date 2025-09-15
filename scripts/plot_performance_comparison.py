import json
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


def main():
    json_path = Path(__file__).parent / "performance_comparison.nljson"
    with json_path.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    fig, ax = plt.subplots()
    for method, label in [
        ("cpu", "CPU"),
        ("naive", "GPU (naive)"),
        ("efficient", "GPU (work-efficient)"),
        ("thrust", "GPU (Thrust)"),
    ]:
        n_values = []
        time_values = []
        for entry in data:
            if entry["method"] == method:
                n_values.append(entry["n"])
                time_values.append(entry["time"])
        n_values = np.array(n_values)
        time_values = np.array(time_values)
        sorted_indices = np.argsort(n_values)
        n_values = n_values[sorted_indices]
        time_values = time_values[sorted_indices]
        ax.plot(n_values, time_values, label=label, marker="o")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size")
    ax.set_ylabel("Time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent.parent / "img" / "performance-comparison.png")


if __name__ == "__main__":
    main()
