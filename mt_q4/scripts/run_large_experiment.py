import argparse
import json
from typing import Dict, List

from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_doc_lookup, compose_context, load_hotpotqa
from src.evaluate import evaluate_generation, evaluate_retrieval
from src.generator import AnswerGenerator, set_seed
from src.retrievers import BM25Retriever

def main():
    parser = argparse.ArgumentParser(description="Large-scale RAG experiments with improvements")
    parser.add_argument("--sample-size", type=int, default=1000, help="Larger sample for better statistics")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieve more documents")
    parser.add_argument("--top-n-context", type=int, default=5, help="Use more documents in context")
    parser.add_argument("--max-context-chars", type=int, default=3000, help="Longer context for better answers")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5, help="More beams for better generation")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-report", default="results_large_scale.json")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    print(f"Running large-scale experiment with {args.sample_size} samples...")
    print(f"Config: top-k={args.top_k}, top-n-context={args.top_n_context}")
    print(f"Context size: {args.max_context_chars} chars, Max tokens: {args.max_new_tokens}")
    
    set_seed(args.seed)
    
    # Load data
    print("\nLoading HotpotQA dataset...")
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    
    # Initialize retriever
    print("Initializing BM25 retriever...")
    retriever = BM25Retriever(documents)
    
    # Initialize generator
    print("Loading T5 generator...")
    device = "cuda" if args.use_gpu else "cpu"
    generator = AnswerGenerator(
        model_name="t5-small",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    print(f"Using device: {device}")

    # Retrieval phase
    print("\nPhase 1: Retrieval")
    retrieval_rows: List[Dict] = []
    for example in tqdm(examples, desc="Retrieving"):
        hits = retriever.search(example.question, k=args.top_k)
        retrieval_rows.append(
            {
                "hits": hits,
                "relevant_titles": example.relevant_titles,
                "question": example.question,
            }
        )

    retrieval_metrics = evaluate_retrieval(
        retrieval_rows, doc_lookup=doc_lookup, k=args.top_k
    )

    print("\nRetrieval Results:")
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Generation phase
    print("\nPhase 2: Generation")
    predictions: List[str] = []
    references: List[str] = []
    qualitative: List[Dict] = []

    for example, row in tqdm(
        list(zip(examples, retrieval_rows)), desc="Generating", total=len(examples)
    ):
        doc_ids = [hit.doc_id for hit in row["hits"][: args.top_n_context]]
        context = compose_context(
            doc_lookup, doc_ids, max_chars=args.max_context_chars
        )
        prediction = generator.generate(example.question, context)
        predictions.append(prediction)
        references.append(example.answer)
        
        retrieved_titles = [doc_lookup[idx].title for idx in doc_ids]
        is_faithful = any(title in example.relevant_titles for title in retrieved_titles)
        
        # Save first 20 examples for analysis
        if len(qualitative) < 20:
            qualitative.append(
                {
                    "question": example.question,
                    "reference": example.answer,
                    "prediction": prediction,
                    "context_titles": retrieved_titles,
                    "relevant_titles": list(example.relevant_titles),
                    "is_faithful": is_faithful,
                }
            )

    generation_metrics = evaluate_generation(
        predictions,
        references,
        bertscore_model=args.bertscore_model,
    )

    print("\nGeneration Results:")
    for k, v in generation_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save report
    config = vars(args).copy()
    config.update({"retriever": "bm25", "generator": "t5-small"})

    report = {
        "config": config,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "qualitative_examples": qualitative,
        "statistics": {
            "total_samples": len(examples),
            "faithful_generations": sum(1 for ex in qualitative if ex["is_faithful"]),
            "hallucinated_generations": sum(1 for ex in qualitative if not ex["is_faithful"]),
        }
    }

    with open(args.save_report, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Saved report to: {args.save_report}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
