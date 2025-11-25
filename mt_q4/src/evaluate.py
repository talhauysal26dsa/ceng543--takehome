from typing import Dict, Iterable, List, Sequence, Set
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from .data import Document
from .retrievers import RetrievedDocument

def evaluate_retrieval(
    results: Sequence[Dict],
    doc_lookup: Dict[str, Document],
    k: int,
) -> Dict[str, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    for row in results:
        hits: List[RetrievedDocument] = row["hits"][:k]
        relevant_titles: Set[str] = row["relevant_titles"]
        if not relevant_titles:
            continue
        retrieved_relevant = sum(
            1
            for hit in hits
            if doc_lookup[hit.doc_id].title in relevant_titles
        )
        precisions.append(retrieved_relevant / k)
        recalls.append(retrieved_relevant / len(relevant_titles))
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
    }

def evaluate_generation(
    predictions: Sequence[str],
    references: Sequence[str],
    bertscore_model: str = "distilbert-base-uncased",
) -> Dict[str, float]:
    smoothing = SmoothingFunction().method1
    bleu = corpus_bleu(
        [[ref.split()] for ref in references],
        [pred.split() for pred in predictions],
        smoothing_function=smoothing,
    ) * 100
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        rouge_scores.append(rouge.score(ref, pred)["rougeL"].fmeasure)
    precision, recall, f1 = bert_score(
        predictions, references, lang="en", model_type=bertscore_model, verbose=False
    )
    return {
        "bleu": float(bleu),
        "rougeL_f1": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "bertscore_precision": float(precision.mean()),
        "bertscore_recall": float(recall.mean()),
        "bertscore_f1": float(f1.mean()),
    }

__all__ = ["evaluate_retrieval", "evaluate_generation"]
