# Sentiment Analysis with Bidirectional RNNs

## Proje Açıklaması

Bu proje, **IMDb film yorumları** üzerinde **duygu analizi (sentiment analysis)** yapmak için farklı derin öğrenme mimarilerini karşılaştırır.

### Amaç

Film yorumlarını okuyup "olumlu" veya "olumsuz" diye sınıflandıran modeller geliştirmek ve hangisinin daha iyi çalıştığını görmek.

## Kullanılan Teknolojiler

### Modeller:

1. **Bidirectional LSTM (Çift Yönlü LSTM)**

   - Metni hem soldan sağa, hem sağdan sola okur
   - Uzun bağlamlı bilgiyi hatırlar
   - Örnek: "Bu film harika DEĞİL" cümlesinde "DEĞİL" kelimesinin önemini anlar

2. **Bidirectional GRU (Çift Yönlü GRU)**
   - LSTM'e benzer ama daha basit yapı
   - Daha hızlı eğitilir
   - Daha az parametre kullanır

### Embedding (Kelime Temsilleri):

#### A) Static Embeddings (Sabit):

- **GloVe (Global Vectors for Word Representation)**
- Her kelime sabit bir vektör ile temsil edilir
- Örnek: "good" kelimesi her zaman aynı [0.2, -0.5, 0.8, ...] vektörü

#### B) Contextual Embeddings (Bağlamsal):

- **BERT (Bidirectional Encoder Representations from Transformers)**
- Kelimelerin anlamı cümleye göre değişir
- Örnek:
  - "Bank is closed" → bank = finans kurumu
  - "River bank is beautiful" → bank = nehir kenarı

## 📊 Değerlendirme Metrikleri

1. **Accuracy (Doğruluk):** Kaç tahminin doğru olduğu
2. **Macro F1:** Hem pozitif hem negatif sınıflar için dengeli performans
3. **Convergence Efficiency:** Model ne kadar hızlı öğreniyor


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
