# Question 3: Seq2Seq (Additive Attention) vs Transformer

## 1) Setup

```bash
cd mt_q3
python -m venv venv3
venv3\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 2) Data preparation

`train.py` will download **Multi30k** via HuggingFace `datasets` on first run (internet needed). Set `dataset.name` to `iwslt2014` in config to switch.

## 3) Training

Seq2Seq (Bahdanau):

```bash
python train.py --model seq2seq --config config.yaml
```

Transformer:

```bash
python train.py --model transformer --config config.yaml
```

Overrides:

- `--epochs`, `--batch_size` override config values.
- `--no_cuda` forces CPU.

## 4) Evaluation (BLEU/ROUGE)

```bash
python evaluate.py --model seq2seq --config config.yaml --checkpoint experiments/seq2seq/best.pt
python evaluate.py --model transformer --config config.yaml --checkpoint experiments/transformer/best.pt
```

Outputs are written as `metrics.json` next to the checkpoint.

## 5) Ablation (layers / attention heads)

Config:

```yaml
ablation:
  layers: [2, 4, 6]
  heads: [2, 4, 8]
```

Template runner:

```bash
python ablation.py --config config.yaml
```

(Minimal script; track/report each run manually.)

#
