import json
import subprocess
from pathlib import Path

OUTPUT_PATH = Path("results_bm25_200.json")

def run_experiment() -> Path:
    cmd = [
        "python",
        "-m",
        "src.run_experiments",
        "--sample-size",
        "200",
        "--top-k",
        "5",
        "--top-n-context",
        "3",
        "--seed",
        "42",
        "--save-report",
        str(OUTPUT_PATH),
    ]
    print("\nRunning BM25 + T5-small experiment")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    return OUTPUT_PATH

def print_summary(path: Path) -> None:
    if not path.exists():
        print(f"No report found at {path}")
        return
    with open(path) as f:
        report = json.load(f)

    retrieval = report.get("retrieval_metrics", {})
    generation = report.get("generation_metrics", {})

    print("\nRetrieval metrics")
    for name, value in retrieval.items():
        print(f"{name}: {value:.4f}")

    print("\nGeneration metrics")
    for name, value in generation.items():
        print(f"{name}: {value:.4f}")

def main() -> None:
    path = run_experiment()
    print_summary(path)

if __name__ == "__main__":
    main()
