import argparse
import json
from typing import Dict, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import build_doc_lookup, compose_context, load_hotpotqa
from src.evaluate import evaluate_generation, evaluate_retrieval
from src.generator import set_seed
from src.retrievers import BM25Retriever

class LargeAnswerGenerator:
        def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 128,
        num_beams: int = 5,
        device: str = "cuda",
    ):
        self.device = device
        print(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # FP16 for speed
        ).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        if device == "cuda":
            print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    def generate(self, question: str, context: str) -> str:
        # Optimized prompt for FLAN-T5
        prompt = f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():  # Disable gradients for inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                length_penalty=0.6,
                no_repeat_ngram_size=3,
                temperature=0.7,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Maximum GPU utilization RAG")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n-context", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=4000, help="Longer context for better answers")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=6, help="More beams for better quality")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="google/flan-t5-base", 
                       choices=["t5-base", "google/flan-t5-base", "google/flan-t5-large"],
                       help="Model to use (larger = better but more VRAM)")
    parser.add_argument("--save-report", default="results_max_gpu.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = "cuda"
    print(f"\n{'='*70}")
    print(f"🚀 MAXIMUM GPU UTILIZATION MODE")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: {args.model_name}")
    print(f"Sample size: {args.sample_size}")
    print(f"Top-k: {args.top_k}, Context docs: {args.top_n_context}")
    print(f"Max context: {args.max_context_chars} chars")
    print(f"{'='*70}\n")
    
    torch.cuda.empty_cache()
    set_seed(args.seed)
    
    # Load data
    print("📥 Loading HotpotQA dataset...")
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    
    # Retriever
    print(f"🔍 Initializing BM25 with {len(documents)} documents...")
    retriever = BM25Retriever(documents)
    
    # Generator - larger model
    print(f"\n🤖 Loading {args.model_name}...")
    generator = LargeAnswerGenerator(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    
    print(f"\n{'='*70}")
    print("PHASE 1: RETRIEVAL")
    print(f"{'='*70}")
    
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

    print(f"\n{'='*70}")
    print("PHASE 2: GENERATION (Large Model)")
    print(f"{'='*70}")
    
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
        
        # Monitor VRAM every 100 iterations
        if len(predictions) % 100 == 0:
            current_vram = torch.cuda.memory_allocated() / 1024**2
            if len(predictions) == 100:
                print(f"   VRAM usage after 100 generations: {current_vram:.0f} MB")

    max_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n💾 Peak VRAM usage: {max_vram:.0f} MB ({max_vram/1024:.2f} GB)")

    print("\n📈 Evaluating generation quality...")
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
    
    config = vars(args).copy()
    config.update({
        "retriever": "bm25",
        "generator": args.model_name,
        "device": device,
        "peak_vram_mb": max_vram,
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
    print(f"Model: {args.model_name}")
    print(f"BLEU: {generation_metrics['bleu']:.2f}")
    print(f"BERTScore F1: {generation_metrics['bertscore_f1']:.4f}")
    print(f"Faithful: {faithful_count}/{len(qualitative)} ({faithful_count/len(qualitative)*100:.1f}%)")
    print(f"Peak VRAM: {max_vram:.0f} MB / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"Report: {args.save_report}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
