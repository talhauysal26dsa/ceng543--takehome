import importlib
import sys

REQUIRED = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("datasets", "datasets"),
    ("rank_bm25", "rank-bm25"),
    ("nltk", "nltk"),
    ("rouge_score", "rouge-score"),
    ("bert_score", "bert-score"),
    ("tqdm", "tqdm"),
]

def check(name: str, label: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"[OK] {label}")
        return True
    except ImportError:
        print(f"[MISSING] {label}")
        return False

def main() -> None:
    missing = [label for mod, label in REQUIRED if not check(mod, label)]
    if missing:
        print("\nInstall with: pip install -r requirements.txt")
        sys.exit(1)
    print("\nAll required packages are available.")

if __name__ == "__main__":
    main()
