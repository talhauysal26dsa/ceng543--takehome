import argparse
import json
from pathlib import Path
from typing import Dict

def load_result(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)

def print_summary_table(results: Dict[str, Dict]) -> None:
    print("\n=== Retrieval Performance ===")
    print("| Run | Sample Size | Precision@k | Recall@k |")
    print("|-----|-------------|-------------|----------|")
    for name, data in results.items():
        config = data.get("config", {})
        r = data.get("retrieval_metrics", {})
        sample_size = config.get("sample_size", "N/A")
        k = config.get("top_k", 5)
        print(
            f"| {name:<3} | {sample_size:<11} | {r.get('precision_at_k', 0):<11.4f} | "
            f"{r.get('recall_at_k', 0):<8.4f} | (k={k})"
        )

    print("\n=== Generation Performance ===")
    print("| Run | BLEU  | ROUGE-L | BERTScore |")
    print("|-----|------|---------|-----------|")
    for name, data in results.items():
        g = data.get("generation_metrics", {})
        print(
            f"| {name:<3} | {g.get('bleu', 0):<5.2f} | {g.get('rougeL_f1', 0):<7.4f} | "
            f"{g.get('bertscore_f1', 0):<9.4f} |"
        )

def extract_examples(results: Dict[str, Dict], n: int) -> None:
    print(f"\n=== Qualitative Examples (first {n}) ===")
    for name, data in results.items():
        examples = data.get("qualitative_examples", [])[:n]
        print(f"\nRun: {name} (showing {len(examples)} examples)")
        for idx, row in enumerate(examples, 1):
            print(f"\n{idx}. Question: {row.get('question')}")
            print(f"   Reference: {row.get('reference')}")
            print(f"   Prediction: {row.get('prediction')}")
            print(f"   Context titles: {', '.join(row.get('context_titles', []))}")
            print(f"   Relevant titles: {', '.join(row.get('relevant_titles', []))}")

def print_statistics(results: Dict[str, Dict]) -> None:
    print("\n=== Stats ===")
    for name, data in results.items():
        config = data.get("config", {})
        r = data.get("retrieval_metrics", {})
        g = data.get("generation_metrics", {})
        k = config.get("top_k", 5)
        print(f"\nRun: {name}")
        print(f"Sample size: {config.get('sample_size', 'N/A')}")
        print(f"Top-k: {k}")
        print(f"Seed: {config.get('seed', 'N/A')}")
        print(f"Precision@{k}: {r.get('precision_at_k', 0):.4f}")
        print(f"Recall@{k}: {r.get('recall_at_k', 0):.4f}")
        print(f"BLEU: {g.get('bleu', 0):.2f}")
        print(f"ROUGE-L (F1): {g.get('rougeL_f1', 0):.4f}")
        print(f"BERTScore (F1): {g.get('bertscore_f1', 0):.4f}")

def interplay(results: Dict[str, Dict]) -> None:
    print("\n=== Retrieval vs Generation ===")
    print("| Run | Precision@k | BLEU | |Precision-BLEU/40| |")
    print("|-----|-------------|------|--------------------|")
    for name, data in results.items():
        r = data.get("retrieval_metrics", {})
        g = data.get("generation_metrics", {})
        prec = r.get("precision_at_k", 0)
        bleu = g.get("bleu", 0)
        diff = abs(prec - (bleu / 40.0))
        print(f"| {name:<3} | {prec:<11.4f} | {bleu:<4.2f} | {diff:<18.4f} |")
    print("\nHigher Precision@k generally correlates with higher ROUGE/BERTScore because the context stays on-topic.")
    print("If BLEU lags while Precision@k is good, inspect qualitative examples for truncated or noisy context.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BM25 experiment reports")
    parser.add_argument("files", nargs="+", help="JSON report files")
    parser.add_argument(
        "--examples", type=int, default=5, help="How many qualitative rows to print"
    )
    parser.add_argument(
        "--format",
        choices=["all", "table", "examples", "stats", "interplay"],
        default="all",
    )
    args = parser.parse_args()

    results: Dict[str, Dict] = {}
    for file in args.files:
        path = Path(file)
        if not path.exists():
            print(f"Skipping missing file: {file}")
            continue
        name = path.stem.replace("results_", "")
        results[name.upper()] = load_result(path)

    if not results:
        print("No valid result files found.")
        return

    if args.format in ("all", "table"):
        print_summary_table(results)
    if args.format in ("all", "examples"):
        extract_examples(results, args.examples)
    if args.format in ("all", "stats"):
        print_statistics(results)
    if args.format in ("all", "interplay"):
        interplay(results)

if __name__ == "__main__":
    main()
