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
from src.retrievers import BM25Retriever, DenseRetriever

def run_retriever_experiment(
    retriever, 
    retriever_name: str,
    examples, 
    doc_lookup, 
    generator,
    args
) -> Dict:
        retrieval_rows: List[Dict] = []
    for example in tqdm(examples, desc=f"Retrieving ({retriever_name})"):
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

    predictions: List[str] = []
    references: List[str] = []
    qualitative: List[Dict] = []

    for example, row in tqdm(
        list(zip(examples, retrieval_rows)), 
        desc=f"Generating ({retriever_name})", 
        total=len(examples)
    ):
        doc_ids = [hit.doc_id for hit in row["hits"][: args.top_n_context]]
        context = compose_context(
            doc_lookup, doc_ids, max_chars=args.max_context_chars
        )
        prediction = generator.generate(example.question, context)
        predictions.append(prediction)
        references.append(example.answer)
        
        # Check if any retrieved doc is relevant
        retrieved_titles = [doc_lookup[idx].title for idx in doc_ids]
        is_faithful = any(title in example.relevant_titles for title in retrieved_titles)
        
        qualitative.append(
            {
                "question": example.question,
                "reference": example.answer,
                "prediction": prediction,
                "context_titles": retrieved_titles,
                "relevant_titles": list(example.relevant_titles),
                "is_faithful": is_faithful,  # True if at least one relevant doc was retrieved
            }
        )

    generation_metrics = evaluate_generation(
        predictions,
        references,
        bertscore_model=args.bertscore_model,
    )

    return {
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "qualitative_examples": qualitative[:10],  # Keep more examples for analysis
    }

def main():
    parser = argparse.ArgumentParser(description="Compare BM25 vs Dense retriever")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n-context", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=1500)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2", 
                       help="Sentence-transformer model for dense retriever")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-report", default="retriever_comparison.json")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Load data
    print("Loading HotpotQA dataset...")
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    
    # Initialize generator (shared by both retrievers)
    print("Loading generator...")
    generator = AnswerGenerator(
        model_name="t5-small",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    
    # Initialize retrievers
    print("\nInitializing BM25 retriever...")
    bm25_retriever = BM25Retriever(documents)
    
    print("\nInitializing Dense retriever...")
    dense_retriever = DenseRetriever(documents, model_name=args.dense_model)
    
    # Run experiments
    print("\n" + "="*60)
    print("Running BM25 experiment")
    print("="*60)
    bm25_results = run_retriever_experiment(
        bm25_retriever, "BM25", examples, doc_lookup, generator, args
    )
    
    print("\n" + "="*60)
    print("Running Dense retriever experiment")
    print("="*60)
    dense_results = run_retriever_experiment(
        dense_retriever, "Dense", examples, doc_lookup, generator, args
    )
    
    # Compute comparison statistics
    comparison = {
        "retrieval_improvement": {
            "precision_diff": dense_results["retrieval_metrics"]["precision_at_k"] - 
                            bm25_results["retrieval_metrics"]["precision_at_k"],
            "recall_diff": dense_results["retrieval_metrics"]["recall_at_k"] - 
                         bm25_results["retrieval_metrics"]["recall_at_k"],
        },
        "generation_improvement": {
            "bleu_diff": dense_results["generation_metrics"]["bleu"] - 
                        bm25_results["generation_metrics"]["bleu"],
            "rougeL_diff": dense_results["generation_metrics"]["rougeL_f1"] - 
                          bm25_results["generation_metrics"]["rougeL_f1"],
            "bertscore_f1_diff": dense_results["generation_metrics"]["bertscore_f1"] - 
                                bm25_results["generation_metrics"]["bertscore_f1"],
        }
    }
    
    # Create full report
    report = {
        "config": vars(args),
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "comparison": comparison,
    }
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\nBM25 Retrieval:")
    for k, v in bm25_results["retrieval_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nDense Retrieval:")
    for k, v in dense_results["retrieval_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nBM25 Generation:")
    for k, v in bm25_results["generation_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nDense Generation:")
    for k, v in dense_results["generation_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovement (Dense - BM25):")
    print("  Retrieval:")
    for k, v in comparison["retrieval_improvement"].items():
        print(f"    {k}: {v:+.4f}")
    print("  Generation:")
    for k, v in comparison["generation_improvement"].items():
        print(f"    {k}: {v:+.4f}")
    
    # Save report
    with open(args.save_report, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2, ensure_ascii=False)
    print(f"\nSaved full comparison report to {args.save_report}")

if __name__ == "__main__":
    main()
