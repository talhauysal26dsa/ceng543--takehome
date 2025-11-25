import argparse
import json
from typing import Dict, List
import torch

from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_doc_lookup, compose_context, load_hotpotqa
from src.evaluate import evaluate_generation, evaluate_retrieval
from src.generator import AnswerGenerator, set_seed
from src.retrievers import BM25Retriever

def main():
    parser = argparse.ArgumentParser(description="Optimized RAG with T5-small")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieve 10 docs")
    parser.add_argument("--top-n-context", type=int, default=5, help="Use top 5 in context")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-report", default="results_optimized.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("OPTIMIZED RAG EXPERIMENT (T5-small)")
    print(f"{'='*70}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Device: {device}")
    print(f"Sample size: {args.sample_size}")
    print(f"Top-k: {args.top_k}, Context: {args.top_n_context} docs")
    print(f"{'='*70}\n")
    
    set_seed(args.seed)
    
    # Load data
    print("Loading HotpotQA...")
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    
    # Retriever
    print(f"Initializing BM25 with {len(documents)} documents...")
    retriever = BM25Retriever(documents)
    
    # Generator
    print("Loading T5-small generator...")
    generator = AnswerGenerator(
        model_name="t5-small",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )

    # Retrieval
    print("\n" + "="*70)
    print("PHASE 1: RETRIEVAL")
    print("="*70)
    
    retrieval_rows: List[Dict] = []
    for example in tqdm(examples, desc="Retrieving"):
        hits = retriever.search(example.question, k=args.top_k)
        retrieval_rows.append({
            "hits": hits,
            "relevant_titles": example.relevant_titles,
            "question": example.question,
        })

    retrieval_metrics = evaluate_retrieval(
        retrieval_rows, doc_lookup=doc_lookup, k=args.top_k
    )

    print("\n📊 Retrieval Results:")
    for k, v in retrieval_metrics.items():
        print(f"   {k}: {v:.4f}")

    # Generation
    print("\n" + "="*70)
    print("PHASE 2: GENERATION")
    print("="*70)
    
    predictions: List[str] = []
    references: List[str] = []
    qualitative: List[Dict] = []

    for example, row in tqdm(
        list(zip(examples, retrieval_rows)), 
        desc="Generating",
        total=len(examples)
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
        
        if len(qualitative) < 20:
            qualitative.append({
                "question": example.question,
                "reference": example.answer,
                "prediction": prediction,
                "context_titles": retrieved_titles,
                "relevant_titles": list(example.relevant_titles),
                "is_faithful": is_faithful,
            })

    if device == "cuda":
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    print("\nEvaluating generation quality...")
    generation_metrics = evaluate_generation(
        predictions,
        references,
        bertscore_model=args.bertscore_model,
    )

    print("\n📊 Generation Results:")
    for k, v in generation_metrics.items():
        print(f"   {k}: {v:.4f}")

    faithful_count = sum(1 for ex in qualitative if ex["is_faithful"])
    
    config = vars(args).copy()
    config.update({
        "retriever": "bm25",
        "generator": "t5-small",
        "device": device,
    })

    report = {
        "config": config,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "qualitative_examples": qualitative,
        "statistics": {
            "total_samples": len(examples),
            "faithful_generations": faithful_count,
            "faithful_ratio": faithful_count / len(qualitative) if qualitative else 0,
        }
    }

    with open(args.save_report, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("✅ EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    print(f"BLEU: {generation_metrics['bleu']:.2f}")
    print(f"ROUGE-L: {generation_metrics['rougeL_f1']:.4f}")
    print(f"BERTScore F1: {generation_metrics['bertscore_f1']:.4f}")
    print(f"Recall@{args.top_k}: {retrieval_metrics['recall_at_k']:.4f}")
    print(f"Faithful: {faithful_count}/{len(qualitative)} ({faithful_count/len(qualitative)*100:.1f}%)")
    print(f"Report: {args.save_report}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
