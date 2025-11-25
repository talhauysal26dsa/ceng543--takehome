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
    parser = argparse.ArgumentParser(description="GPU-optimized RAG experiments")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of examples")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieve more docs with GPU speed")
    parser.add_argument("--top-n-context", type=int, default=5, help="Use more context")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-report", default="results_gpu_optimized.json")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation (adjust for VRAM)")
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Running on CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear cache
        torch.cuda.empty_cache()

    print(f"\nConfiguration:")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Top-k retrieval: {args.top_k}")
    print(f"  Context docs: {args.top_n_context}")
    print(f"  Max context: {args.max_context_chars} chars")
    print(f"  Generation batch size: {args.batch_size}")
    print(f"  Device: {device}\n")
    
    set_seed(args.seed)
    
    # Load data
    print("Loading HotpotQA dataset...")
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    
    # Initialize retriever
    print(f"Initializing BM25 retriever with {len(documents)} documents...")
    retriever = BM25Retriever(documents)
    
    # Initialize generator with GPU
    print("Loading T5-small generator on GPU...")
    generator = AnswerGenerator(
        model_name="t5-small",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    
    if device == "cuda":
        print(f"Initial VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # Retrieval phase
    print("\n" + "="*60)
    print("PHASE 1: RETRIEVAL")
    print("="*60)
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

    print("\n📊 Retrieval Results:")
    for k, v in retrieval_metrics.items():
        print(f"   {k}: {v:.4f}")

    # Generation phase with batching
    print("\n" + "="*60)
    print("PHASE 2: GENERATION (GPU-accelerated)")
    print("="*60)
    
    predictions: List[str] = []
    references: List[str] = []
    qualitative: List[Dict] = []

    # Process in batches for memory efficiency
    for i in tqdm(range(0, len(examples), args.batch_size), desc="Generating (batched)"):
        batch_examples = examples[i:i + args.batch_size]
        batch_rows = retrieval_rows[i:i + args.batch_size]
        
        for example, row in zip(batch_examples, batch_rows):
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
        
        # Clear cache periodically
        if device == "cuda" and i % (args.batch_size * 10) == 0:
            torch.cuda.empty_cache()

    if device == "cuda":
        print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    print("\nEvaluating generation quality...")
    generation_metrics = evaluate_generation(
        predictions,
        references,
        bertscore_model=args.bertscore_model,
    )

    print("\n📊 Generation Results:")
    for k, v in generation_metrics.items():
        print(f"   {k}: {v:.4f}")

    # Statistics
    faithful_count = sum(1 for ex in qualitative if ex["is_faithful"])
    
    # Save report
    config = vars(args).copy()
    config.update({
        "retriever": "bm25",
        "generator": "t5-small",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    })

    report = {
        "config": config,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "qualitative_examples": qualitative,
        "statistics": {
            "total_samples": len(examples),
            "faithful_generations": faithful_count,
            "hallucinated_generations": len(qualitative) - faithful_count,
            "faithful_ratio": faithful_count / len(qualitative) if qualitative else 0,
        }
    }

    with open(args.save_report, "w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Experiment completed!")
    print(f"✓ Faithful: {faithful_count}/{len(qualitative)} ({faithful_count/len(qualitative)*100:.1f}%)")
    print(f"✓ Report saved: {args.save_report}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
