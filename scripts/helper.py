import subprocess
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).parent.parent.absolute()
MEASURE_TIME_EXE = ROOT_DIR / "build" / "bin" / "Release" / "measure_time.exe"
assert MEASURE_TIME_EXE.is_file()


def measure_time(
    operation: Literal["scan", "compact"],
    implementation: Literal["cpu", "naive", "efficient", "thrust"],
    input_size: int,
    block_size: int = -1,
) -> float:
    result = subprocess.run(
        [
            str(MEASURE_TIME_EXE),
            operation,
            implementation,
            str(input_size),
            str(block_size),
        ],
        capture_output=True,
        cwd=ROOT_DIR,
        check=True,
        encoding="utf-8",
        text=True,
    )
    return float(result.stdout)
