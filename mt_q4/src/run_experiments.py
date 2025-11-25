import argparse
import json
from typing import Dict, List

from tqdm import tqdm

from .data import build_doc_lookup, compose_context, load_hotpotqa
from .evaluate import evaluate_generation, evaluate_retrieval
from .generator import AnswerGenerator, set_seed
from .retrievers import BM25Retriever

def main():
    parser = argparse.ArgumentParser(description="HotpotQA RAG experiments (BM25 + T5-small)")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n-context", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-report", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    documents, examples = load_hotpotqa(
        sample_size=args.sample_size, split=args.split, seed=args.seed
    )
    doc_lookup = build_doc_lookup(documents)
    retriever_name = "bm25"
    generator_name = "t5-small"

    retriever = BM25Retriever(documents)
    generator = AnswerGenerator(
        model_name=generator_name,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

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
        
        # Check if any retrieved doc is relevant (for faithful vs hallucinated analysis)
        retrieved_titles = [doc_lookup[idx].title for idx in doc_ids]
        is_faithful = any(title in example.relevant_titles for title in retrieved_titles)
        
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

    config = vars(args).copy()
    config.update(
        {
            "retriever": retriever_name,
            "generator": generator_name,
        }
    )

    report = {
        "config": config,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "qualitative_examples": qualitative[:5],
    }

    print("\nRetrieval metrics")
    for k, v in retrieval_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nGeneration metrics")
    for k, v in generation_metrics.items():
        print(f"{k}: {v:.4f}")

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as fout:
            json.dump(report, fout, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {args.save_report}")

if __name__ == "__main__":
    main()
