

## 🚀 Kurulum

1. Sanal ortam oluştur:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Paketleri yükle:

```bash
pip install -r requirements.txt
```

3. IMDb verisini indir:

```bash
python utils/download_data.py
```

4. GloVe embedding'lerini indir:

```bash
python utils/download_glove.py
```

## 🎓 Kullanım

### Tüm modelleri eğit:

```bash
python train.py --model all --embedding all
```

### Belirli bir model eğit:

```bash
# LSTM + GloVe
python train.py --model lstm --embedding glove

# GRU + BERT
python train.py --model gru --embedding bert
```

### Sonuçları değerlendir:

```bash
python evaluate.py --experiment_dir experiments/
```

## 📈 Beklenen Sonuçlar

Karşılaştıracağımız 4 kombinasyon:

1. BiLSTM + GloVe
2. BiLSTM + BERT
3. BiGRU + GloVe
4. BiGRU + BERT

Her biri için:

- Accuracy ve F1 skorları
- Eğitim süresi
- Epoch başına öğrenme hızı

## 🔍 Temel Kavramlar

### LSTM vs GRU

- LSTM: 3 kapı (forget, input, output) - daha güçlü ama yavaş
- GRU: 2 kapı (reset, update) - daha basit ama hızlı

### Static vs Contextual Embeddings

- Static (GloVe): Önceden eğitilmiş, sabit vektörler
- Contextual (BERT): Cümleye göre değişen, dinamik vektörler

### Bidirectional (Çift Yönlü)

- Metni hem ileriye hem geriye doğru okur
- Daha iyi bağlam anlayışı sağlar

## 📚 Kaynaklar

- IMDb Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- GloVe: https://nlp.stanford.edu/projects/glove/
- BERT: https://huggingface.co/bert-base-uncased

## 👨‍💻 Geliştirici Notları

Bu proje educational amaçlıdır ve sequence classification task'ları için farklı yaklaşımları karşılaştırmayı hedefler.
