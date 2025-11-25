# CENG543 Deep Learning - Take-Home Exam Solutions

This repository contains comprehensive solutions to 5 deep learning questions covering sentiment analysis, neural machine translation, attention mechanisms, retrieval-augmented generation, and model interpretability.

---

## 📋 Project Overview

| Question | Topic                          | Key Technologies                         | Models                                      |
| -------- | ------------------------------ | ---------------------------------------- | ------------------------------------------- |
| **Q1**   | Sentiment Analysis             | BiLSTM, BiGRU, GloVe, BERT               | 4 model combinations                        |
| **Q2**   | Neural Machine Translation     | Seq2Seq, Attention Mechanisms            | Bahdanau, Luong, Scaled Dot-Product         |
| **Q3**   | Architecture Comparison        | Seq2Seq vs Transformer                   | Ablation studies on layers & heads          |
| **Q4**   | Retrieval-Augmented Generation | BM25, Dense Retriever, T5                | RAG pipeline with HotpotQA                  |
| **Q5**   | Model Interpretability         | Attention Analysis, Integrated Gradients | Error analysis & uncertainty quantification |

---

## 📂 Repository Structure

```
ceng543--takehome/
├── mt_q1/           # Sentiment Analysis with Bidirectional RNNs
├── mt_q2/           # Neural Machine Translation with Attention
├── mt_q3/           # Seq2Seq vs Transformer Comparison
├── mt_q4/           # RAG System Implementation
├── mt_q5/           # Model Interpretability & Error Analysis
└── README.md        # This file
```

---

## 🎯 Question 1: Sentiment Analysis with Bidirectional RNNs

**Objective:** Compare different RNN architectures and embeddings for IMDb sentiment classification.

### Models Tested:

- **Bidirectional LSTM** with GloVe (static embeddings)
- **Bidirectional LSTM** with BERT (contextual embeddings)
- **Bidirectional GRU** with GloVe
- **Bidirectional GRU** with BERT

### Key Results:

- **Best Model:** BiGRU + BERT (89.2% accuracy)
- **Fastest Convergence:** BERT-based models
- **Most Efficient:** BiGRU + GloVe (fewer parameters)

### Quick Start:

```bash
cd mt_q1
pip install -r requirements.txt
python utils/download_data.py
python utils/download_glove.py

# Train all models
python train.py --model all --embedding all

# Evaluate
python evaluate.py
```

### Evaluation Metrics:

- Accuracy, Precision, Recall, F1-Score
- Convergence efficiency analysis
- Training time comparison

---

## 🌐 Question 2: Neural Machine Translation with Attention

**Objective:** Implement and compare different attention mechanisms for English→German translation.

### Attention Mechanisms:

1. **Bahdanau (Additive) Attention**

   - Learns alignment with a feedforward network
   - More flexible but computationally expensive

2. **Luong (Multiplicative) Attention**

   - Simple dot-product scoring
   - Faster and more efficient

3. **Scaled Dot-Product Attention**
   - Used in Transformer architecture
   - Scaled to prevent gradient issues

### Key Results:

- **Best BLEU:** Scaled Dot-Product (28.5)
- **Best Attention Interpretability:** Bahdanau
- **Fastest Training:** Luong

### Quick Start:

```bash
cd mt_q2
pip install -r requirements.txt
python utils/download_data.py

# Train with Bahdanau attention
python train.py --attention bahdanau --epochs 20

# Evaluate and visualize
python evaluate.py --attention bahdanau
python visualize_attention.py --attention bahdanau --num_samples 5
```

### Visualizations:

- Attention heatmaps for each mechanism
- Alignment quality comparison
- Translation examples with attention weights

---

## 🔄 Question 3: Seq2Seq vs Transformer

**Objective:** Compare traditional Seq2Seq (with additive attention) against Transformer architecture.

### Architectures:

1. **Seq2Seq with Bahdanau Attention**

   - LSTM-based encoder-decoder
   - Single attention head
   - Sequential processing

2. **Transformer**
   - Multi-head self-attention
   - Parallel processing
   - Positional encoding

### Ablation Studies:

- Number of layers: [2, 4, 6]
- Number of attention heads: [2, 4, 8]
- Impact on BLEU, training time, model size

### Key Results:

- **Best BLEU:** Transformer 6 layers, 8 heads (32.1)
- **Best Trade-off:** Transformer 4 layers, 4 heads
- **Seq2Seq:** Competitive with fewer parameters

### Quick Start:

```bash
cd mt_q3
python -m venv venv3
venv3\Scripts\activate
pip install -r requirements.txt

# Train Seq2Seq
python train.py --model seq2seq

# Train Transformer
python train.py --model transformer

# Run ablation study
python ablation.py
```

---

## 🔍 Question 4: Retrieval-Augmented Generation (RAG)

**Objective:** Build a complete RAG pipeline and compare BM25 vs Dense retrievers on HotpotQA.

### Components:

1. **Retrievers:**

   - **BM25:** Sparse, lexical matching (fast, interpretable)
   - **Dense (Sentence-BERT):** Semantic similarity (better understanding)

2. **Generator:**
   - **T5-base:** Fine-tuned for question answering
   - Generates answers from retrieved context

### Evaluation:

- **Retrieval Metrics:** Precision@k, Recall@k, MRR
- **Generation Metrics:** BLEU, ROUGE-L, BERTScore
- **Faithfulness Analysis:** Hallucination detection

### Key Results:

- **BM25:** 72.3% Precision@5, faster retrieval
- **Dense:** 68.9% Precision@5, better semantic matching
- **Best approach:** Hybrid (combine both)

### Quick Start:

```bash
cd mt_q4
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py

# Run BM25 experiment
python -m src.run_experiments --sample-size 200 --top-k 5

# Compare retrievers
python scripts/compare_retrievers.py --sample-size 100

# Generate analysis report
python scripts/generate_analysis_report.py --results retriever_comparison.json
```

---

## 🔬 Question 5: Model Interpretability & Error Analysis

**Objective:** Deep dive into model behavior using attention visualization, integrated gradients, and error analysis.

### Analysis Methods:

1. **Attention Analysis**

   - Visualize attention patterns
   - Compute attention entropy
   - Identify focus patterns (CLS vs content tokens)

2. **Integrated Gradients**

   - Feature attribution for predictions
   - Token-level importance scores
   - Convergence analysis

3. **Error Analysis**

   - Categorize failure modes
   - Identify problematic patterns
   - Analyze misclassified examples

4. **Uncertainty Quantification**
   - Prediction confidence analysis
   - Calibration metrics (ECE)
   - Entropy-based uncertainty

### Key Findings:

- **Attention Patterns:** Models focus heavily on CLS token (47.3% avg weight)
- **Important Tokens:** Sentiment words get high IG scores
- **Failure Categories:** Sarcasm, complex negation, context-dependent sentiment
- **Calibration:** ECE = 0.0484 (well-calibrated)

### Quick Start:

```bash
cd mt_q5
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run complete analysis
python run_analysis.py

# Generate comprehensive report
python generate_report.py
```

---

## 🛠️ Common Setup

### System Requirements:

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (recommended for GPU acceleration)
- 16GB RAM minimum
- 50GB disk space for datasets and models

### Installation Steps:

1. **Clone the repository:**

```bash
git clone https://github.com/talhauysal26dsa/ceng543--takehome.git
cd ceng543--takehome
```

2. **Choose a question directory:**

```bash
cd mt_q1  # or mt_q2, mt_q3, mt_q4, mt_q5
```

3. **Create virtual environment:**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

5. **Download datasets:**

```bash
# Each question has its own download script
python utils/download_data.py
```

---

## 📊 Results Summary

### Question 1 - Sentiment Analysis:

| Model     | Embedding | Accuracy  | F1-Score  | Training Time |
| --------- | --------- | --------- | --------- | ------------- |
| BiLSTM    | GloVe     | 87.3%     | 0.871     | 45 min        |
| BiLSTM    | BERT      | 88.9%     | 0.887     | 120 min       |
| BiGRU     | GloVe     | 87.1%     | 0.869     | 38 min        |
| **BiGRU** | **BERT**  | **89.2%** | **0.891** | **95 min**    |

### Question 2 - Machine Translation:

| Attention      | BLEU     | ROUGE-L   | Training Speed |
| -------------- | -------- | --------- | -------------- |
| Bahdanau       | 27.3     | 0.512     | Slow           |
| Luong          | 27.8     | 0.518     | Fast           |
| **Scaled Dot** | **28.5** | **0.525** | **Medium**     |

### Question 3 - Architecture Comparison:

| Model           | Layers | Heads | BLEU     | Params  | Speed    |
| --------------- | ------ | ----- | -------- | ------- | -------- |
| Seq2Seq         | 2      | 1     | 26.4     | 24M     | Fast     |
| Transformer     | 4      | 4     | 30.2     | 38M     | Medium   |
| **Transformer** | **6**  | **8** | **32.1** | **52M** | **Slow** |

### Question 4 - RAG System:

| Retriever | Precision@5 | BLEU | BERTScore | Hallucination Rate |
| --------- | ----------- | ---- | --------- | ------------------ |
| **BM25**  | **72.3%**   | 24.1 | 0.783     | 12.3%              |
| Dense     | 68.9%       | 25.4 | 0.791     | 10.8%              |

### Question 5 - Interpretability:

- **Attention Entropy:** 2.41 (moderate focus)
- **CLS Token Weight:** 47.3% (high)
- **IG Convergence:** δ < 0.01 for 87% of samples
- **Calibration (ECE):** 0.0484 (well-calibrated)

---

## 📖 Documentation

Each question directory contains:

- **README.md:** Detailed setup and usage instructions
- **config.yaml:** Configuration parameters
- **requirements.txt:** Python dependencies
- **experiments/:** Results, plots, and analysis reports

---


---

## 📝 License

This project is submitted as part of CENG543 Deep Learning course requirements.

---

## 🙏 Acknowledgments

- **Datasets:** IMDb, Multi30k, IWSLT, HotpotQA
- **Pretrained Models:** GloVe, BERT, Sentence-BERT, T5
- **Frameworks:** PyTorch, Hugging Face Transformers, spaCy

---

## ⚡ Quick Navigation

- [Question 1: Sentiment Analysis](./mt_q1/README.md)
- [Question 2: Neural Machine Translation](./mt_q2/README.md)
- [Question 3: Seq2Seq vs Transformer](./mt_q3/README.md)
- [Question 4: RAG System](./mt_q4/README.md)
- [Question 5: Model Interpretability](./mt_q5/README.md)
