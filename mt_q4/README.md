# ceng543 mt_q4 — RAG System with BM25 and Dense Retrievers

Complete RAG (Retrieval-Augmented Generation) pipeline on HotpotQA with:

## Setup

1. Create/activate a virtual env: `python -m venv .venv && .venv\\Scripts\\activate`
2. Install deps: `pip install -r requirements.txt`
3. Download HotpotQA: `python scripts/download_data.py`

## Quick Start

### 1. Run BM25 experiment only:

```bash
python -m src.run_experiments --sample-size 200 --top-k 5 --top-n-context 3 --save-report results_bm25.json
```

### 2. Compare BM25 vs Dense Retriever:

```bash
python scripts/compare_retrievers.py --sample-size 100 --top-k 5 --save-report retriever_comparison.json
```

This will:

- Run experiments with both BM25 and Sentence-BERT retrievers
- Compare retrieval metrics (Precision@k, Recall@k)
- Compare generation metrics (BLEU, ROUGE-L, BERTScore)
- Save a comprehensive comparison report

### 3. Generate Detailed Analysis Report:

```bash
python scripts/generate_analysis_report.py --results retriever_comparison.json --output ANALYSIS_REPORT.md
```

This creates a detailed markdown report including:

- Retriever performance comparison
- Faithful vs. hallucinated generation examples
- Correlation analysis between retrieval quality and generation quality
- Recommendations for improvement

## Components

### Retrievers (`src/retrievers.py`)

- **BM25Retriever:** Sparse retriever using BM25 algorithm (lexical matching)
- **DenseRetriever:** Dense retriever using Sentence-BERT embeddings (semantic similarity)

### Generator (`src/generator.py`)

- T5-small model with beam search for answer generation

### Data (`src/data.py`)

- Loads and preprocesses HotpotQA dataset
- Builds document index and context strings

### Evaluation (`src/evaluate.py`)

- **Retrieval metrics:** Precision@k, Recall@k
- **Generation metrics:** BLEU, ROUGE-L, BERTScore
- Qualitative examples with faithful/hallucinated labels

### Scripts

- `src/run_experiments.py`: Run RAG experiments with a single retriever
- `scripts/compare_retrievers.py`: Compare BM25 vs Dense retriever
- `scripts/generate_analysis_report.py`: Generate comprehensive analysis markdown report

## Expected Results

Based on a 200-sample experiment:

### BM25 Retriever

- Precision@5: ~0.25
- Recall@5: ~0.62
- BLEU: ~2.0
- BERTScore F1: ~0.72

### Dense Retriever (Sentence-BERT)

- Generally higher recall due to semantic similarity
- May retrieve more relevant documents but with lower precision
- Generation quality varies based on retrieved context quality

## Key Findings

1. **Retrieval Quality Matters:** Higher Recall@k strongly correlates with better generation quality (BERTScore)
2. **Faithful vs Hallucinated:** When relevant documents are retrieved, generations are more factually accurate
3. **Trade-offs:** BM25 excels at exact term matching (good for factual queries), while dense retrievers capture semantic similarity (good for paraphrased queries)

## Configuration Options

- `--sample-size`: Number of examples to evaluate (default: 50)
- `--top-k`: Number of documents to retrieve (default: 5)
- `--top-n-context`: Number of top documents to use for generation (default: 3)
- `--max-context-chars`: Maximum context length for generator (default: 1500)
- `--seed`: Random seed for reproducibility (default: 42)
- `--dense-model`: Sentence-transformer model for dense retriever (default: "all-MiniLM-L6-v2")

## Notes

- Seeded runs keep subsampling and generation deterministic
- Dense retriever encoding may take time on first run
- Context length affects both retrieval quality and generation
- For best results, compare both retrievers and analyze the report
