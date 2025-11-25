# Seq2Seq (Bahdanau Attention) - Experiment Report

**Model:** Seq2Seq with Additive (Bahdanau) Attention  
**Dataset:** Multi30k (EN→DE)  
**Location:** `experiments/seq2seq/`  
**Status:** ✅ Completed

---

## 📋 Configuration

```yaml
Architecture:
  Model Type: Seq2Seq
  Encoder: BiLSTM (2 layers, hidden_dim=512)
  Decoder: LSTM (2 layers, hidden_dim=512)
  Attention: Additive (Bahdanau-style, dim=512)
  Embedding: 256-dim, trainable
  Dropout: 0.3

Training:
  Batch Size: 128
  Learning Rate: 0.001
  Optimizer: Adam
  Epochs: 20
  Teacher Forcing: 0.5
  Gradient Clip: 1.0
  Early Stopping: 5 patience

Dataset:
  Name: multi30k
  Source: English (en)
  Target: German (de)
  Min Freq: 2
  Max Length: 50
```

---

## 📊 Final Results

### Translation Quality Metrics

| Metric      | Score     | Quality Level  |
| ----------- | --------- | -------------- |
| **BLEU**    | **30.05** | Orta-İyi arası |
| **ROUGE-1** | **0.632** | İyi            |
| **ROUGE-2** | **0.415** | Orta           |
| **ROUGE-L** | **0.608** | İyi            |

**Evaluation Details:**

- Test Samples: 1000
- Beam Search: Enabled (size=5)
- Max Decode Length: 50

---

## 📈 Training Progress

### Loss Curves

| Epoch | Train Loss | Val Loss  | Time (s) | GPU (MB) |
| ----- | ---------- | --------- | -------- | -------- |
| 1     | 4.848      | 4.520     | 161.0    | 3285     |
| 5     | 2.513      | 3.155     | 160.1    | 3288     |
| 10    | 1.817      | 3.142     | 159.7    | 3289     |
| 11    | 1.742      | **3.033** | 161.8    | 3284     |
| 15    | 1.514      | 3.194     | 165.3    | 3292     |
| 20    | 1.335      | 3.179     | 158.8    | 3289     |

**Best Model:** Epoch 11 (val_loss = 3.033)

### Training Summary

```
Total Epochs: 20
Best Epoch: 11
Training Time: ~53 minutes
Average Epoch Time: ~160 seconds
GPU Memory (peak): ~3292 MB
GPU Memory (avg): ~3288 MB
```

---

## 🔍 Analysis

### Strengths

1. ✅ **Solid Baseline:** BLEU 30.05 is respectable for Seq2Seq
2. ✅ **Stable Training:** Consistent epoch times (~160s)
3. ✅ **Good ROUGE-1:** Strong word-level overlap (0.632)
4. ✅ **Established Architecture:** Well-understood, debuggable

### Weaknesses

1. ❌ **Early Convergence:** Best model at epoch 11, then plateaued
2. ❌ **Overfitting:** Val loss increased after epoch 11 (3.033 → 3.179)
3. ❌ **Slow Training:** ~160s per epoch (4x slower than Transformer)
4. ❌ **High Memory:** 3.3GB GPU memory (76% more than Transformer)
5. ❌ **Sequential Processing:** Cannot parallelize across time steps

### Convergence Pattern

```
Phase 1 (Epochs 1-5):   Rapid descent (4.848 → 2.513 train)
Phase 2 (Epochs 6-10):  Slowdown (2.513 → 1.817 train)
Phase 3 (Epochs 11-20): Plateau (val ~3.03, train keeps dropping)
```

**Diagnosis:** Model hit local minimum early, subsequent training just memorizes training set (overfitting).

---

## 🎯 Performance Breakdown

### BLEU: 30.05

- **Interpretation:** "Orta" kalite çeviriler
- **Comparison:** Baseline sistemler için iyi
- **Limitation:** Modern SOTA'dan (~40+ BLEU) uzak

### ROUGE Scores

- **ROUGE-1 (0.632):** İyi unigram overlap → Kelime seçimi doğru
- **ROUGE-2 (0.415):** Orta bigram overlap → Phrase-level zayıflık
- **ROUGE-L (0.608):** İyi LCS → Genel akıcılık korunuyor

**Gap Analysis:**

```
ROUGE-1 - ROUGE-2 = 0.632 - 0.415 = 0.217
```

Büyük fark → Model bireysel kelimeleri iyi seçiyor ama kelime sırasında/kombinasyonlarında zorluk yaşıyor.

---

## 🔬 Technical Details

### Model Capacity

```
Embedding:     vocab_size × 256 × 2 (src + tgt)
Encoder LSTM:  2 × (4 × (256+512) × 512) ≈ 3.1M params
Decoder LSTM:  2 × (4 × (256+512) × 512) ≈ 3.1M params
Attention:     (512+512) × 512 ≈ 524K params
Output Layer:  512 × vocab_size

Total: ~7-8M parameters (vocab-dependent)
```

### Why Slow?

1. **Sequential LSTM:** Each time step depends on previous
2. **Limited Parallelism:** Only batch dimension can be parallelized
3. **Memory Bandwidth:** Hidden state transfers bottleneck
4. **Attention Computation:** Done for each decoder step separately

### Why Overfit?

1. **High Capacity:** 8M params for 29K training examples
2. **Insufficient Regularization:** dropout=0.3 not enough
3. **Teacher Forcing:** 0.5 ratio might be too high (model relies on ground truth)
4. **No Label Smoothing:** Encourages overconfidence

---

## 💡 Insights

### What Worked

- Bahdanau attention effectively bridges encoder-decoder
- Dropout 0.3 provides some regularization
- Gradient clipping prevents explosion
- Teacher forcing helps initial training

### What Didn't Work

- Early stopping patience 5 → Could be 3 (best at epoch 11, trained to 20)
- Dropout 0.3 → Should be 0.4-0.5 for this capacity
- Learning rate 0.001 → Could try 0.0005 with warmup
- No learning rate decay until plateau → Too late

---

## 🔄 Comparison with Transformer

| Aspect           | Seq2Seq                | Transformer               | Verdict                    |
| ---------------- | ---------------------- | ------------------------- | -------------------------- |
| BLEU             | 30.05                  | 32.29                     | Transformer +7.45% ✅      |
| Training Speed   | ~160s/epoch            | ~38s/epoch                | Transformer 4.2x faster ✅ |
| GPU Memory       | ~3289 MB               | ~1871 MB                  | Transformer 43% less ✅    |
| Convergence      | Epoch 11 (plateau)     | Epoch 14 (stable)         | Transformer better ✅      |
| Overfitting      | Yes (val ↑ after 11)   | Minimal                   | Transformer more robust ✅ |
| Interpretability | Black box hidden state | Attention weights visible | Transformer better ✅      |

**Conclusion:** Transformer superior in every metric.

---

## 🎓 Lessons Learned

### Architecture

- ❌ Recurrent networks too slow for production
- ❌ Sequential processing limits parallelism
- ❌ Hidden state bottleneck hurts long sequences
- ✅ Attention mechanism still valuable (used in Transformer too)

### Training

- ⚠️ Need stronger regularization (dropout, label smoothing)
- ⚠️ Early stopping patience could be lower (3 instead of 5)
- ⚠️ Learning rate warmup might help
- ⚠️ Teacher forcing schedule could be explored (start high, decay)

### Evaluation

- ✅ Beam search helps significantly (vs greedy)
- ✅ BLEU correlates with human judgment
- ✅ ROUGE provides complementary insights

---

## 🔮 Potential Improvements

### If Continuing with Seq2Seq:

**Short-term Fixes:**

1. Increase dropout: 0.3 → 0.4 or 0.5
2. Add label smoothing: ε = 0.1
3. Reduce early stopping patience: 5 → 3
4. Try learning rate warmup: 5 epoch linear warmup

**Medium-term Enhancements:** 5. Scheduled teacher forcing: Start 1.0, decay to 0.3 6. Layer normalization in LSTM 7. Weight tying (embedding ↔ output layer) 8. Variational dropout (same mask across timesteps)

**Long-term Exploration:** 9. Bidirectional decoder (non-autoregressive) 10. Pretrained encoder (BERT-style) 11. Mixture of experts 12. But honestly... just use Transformer 😅

---

## 📁 Files Generated

```
experiments/seq2seq/
├── best.pt              # Checkpoint at epoch 11 (val_loss=3.033)
├── last_20.pt           # Final epoch checkpoint
├── config.yaml          # Training configuration snapshot
├── history.json         # Full training history (loss, time, GPU)
└── metrics.json         # BLEU/ROUGE evaluation results
```

---

## 🎯 Recommendations

### For Production:

**❌ NOT RECOMMENDED**

- Use Transformer instead (better in every way)
- If forced to use RNN, try:
  - Reduce model size (1 layer, hidden=256)
  - Stronger regularization
  - Longer training with better LR schedule

### For Research:

**✅ GOOD BASELINE**

- Useful as comparison point
- Demonstrates RNN limitations
- Shows importance of architecture choice

### For Learning:

**✅ EDUCATIONAL VALUE**

- Good for understanding attention mechanisms
- Clear example of overfitting
- Motivates Transformer design choices

---

## 📊 Final Verdict

**Grade: B-**

**Justification:**

- Decent BLEU for Seq2Seq (30.05)
- Properly trained and evaluated
- Clear overfitting issues
- Slow and memory-heavy
- Outperformed by Transformer in every metric

**Recommendation:** Use as baseline, deploy Transformer for production.

---

**Report Generated:** 23 Kasım 2025  
**Checkpoint:** `experiments/seq2seq/best.pt`  
**Status:** Archived (Transformer preferred)
