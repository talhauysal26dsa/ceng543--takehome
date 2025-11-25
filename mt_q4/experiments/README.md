# Experiments Directory

Reports saved here are JSON files from the single BM25 + T5-small run.

Example:

```json
{
  "config": {
    "retriever": "bm25",
    "generator": "t5-small",
    "sample_size": 200,
    "top_k": 5,
    "seed": 42
  },
  "retrieval_metrics": {
    "precision_at_k": 0.4200,
    "recall_at_k": 0.6100
  },
  "generation_metrics": {
    "bleu": 28.45,
    "rougeL_f1": 0.5234,
    "bertscore_f1": 0.8567
  },
  "qualitative_examples": [...]
}
```

Naming: `results_bm25_<sample_size>.json` (e.g., `results_bm25_200.json`).  
These files are ignored by git; only this README stays in the folder.
