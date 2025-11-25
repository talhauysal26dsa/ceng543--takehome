---


## Quick Start

### 1. Setup

```bash
cd mt_q2
pip install -r requirements.txt
python utils/download_data.py
```

### 2. Train Models

```bash
# Bahdanau Attention
python train.py --attention bahdanau --epochs 20

# Luong Attention
python train.py --attention luong --epochs 20

# Scaled Dot-Product Attention
python train.py --attention scaled_dot --epochs 20
```

### 3. Evaluate

```bash
python evaluate.py --attention bahdanau
python evaluate.py --attention luong
python evaluate.py --attention scaled_dot
```

### 4. Visualize Attention

```bash
python visualize_attention.py --attention bahdanau --num_samples 5
python visualize_attention.py --attention luong --num_samples 5
python visualize_attention.py --attention scaled_dot --num_samples 5
```

### 5. Analyze Attention

```bash
python analyze_attention.py
```

---

## Hyperparameters (Reproducibility)

All hyperparameters are defined in `config.yaml`:

```yaml
# Random Seeds
seed: 42
torch_seed: 42
numpy_seed: 42

# Model Architecture
embedding_dim: 256
hidden_dim: 512
num_layers: 2
dropout: 0.3

# Training
batch_size: 128
learning_rate: 0.001
epochs: 20
gradient_clip: 1.0
```

## References

1. Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
2. Luong et al. (2015) - "Effective Approaches to Attention-based Neural Machine Translation"
3. Vaswani et al. (2017) - "Attention Is All You Need"

---

## Author

Machine Translation Project - CENG543
