

from pathlib import Path
from datasets import load_dataset
import yaml


def download_multi30k(save_dir="data"):
    
    print("=" * 70)
    print("[INFO] Multi30k Dataset indiriliyor...")
    print("=" * 70)
    
    # Dizini oluştur
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        
        print("\n[INFO] Hugging Face 'datasets' arayüzü kullanılıyor...")
        hf_dataset = load_dataset("bentrevett/multi30k")

        print("\n[STEP 1] Training set yükleniyor...")
        train_data = [(item["en"], item["de"]) for item in hf_dataset["train"]]
        print(f"   [OK] {len(train_data)} eğitim cümlesi yüklendi")
        
        print("\n[STEP 2] Validation set yükleniyor...")
        val_data = [(item["en"], item["de"]) for item in hf_dataset["validation"]]
        print(f"   [OK] {len(val_data)} validation cümlesi yüklendi")
        
        print("\n[STEP 3] Test set yükleniyor...")
        test_data = [(item["en"], item["de"]) for item in hf_dataset["test"]]
        print(f"   [OK] {len(test_data)} test cümlesi yüklendi")
        
        # Örnek veriyi göster
        print("\n" + "=" * 70)
        print("[INFO] Örnek Veriler:")
        print("=" * 70)
        
        for i in range(min(3, len(train_data))):
            src, tgt = train_data[i]
            print(f"\nSource EN #{i+1}:")
            print(f"   {src}")
            print("Target DE:")
            print(f"   {tgt}")
        
        # İstatistikler
        print("\n" + "=" * 70)
        print("[INFO] Dataset İstatistikleri:")
        print("=" * 70)
        
        # Ortalama cümle uzunlukları
        train_src_lens = [len(src.split()) for src, _ in train_data]
        train_tgt_lens = [len(tgt.split()) for _, tgt in train_data]
        
        print("\n[STATS] Ortalama Kelime Sayısı (Train):")
        print(f"   İngilizce: {sum(train_src_lens) / len(train_src_lens):.1f} kelime")
        print(f"   Almanca:   {sum(train_tgt_lens) / len(train_tgt_lens):.1f} kelime")
        
        print("\n[STATS] Maksimum Kelime Sayısı (Train):")
        print(f"   İngilizce: {max(train_src_lens)} kelime")
        print(f"   Almanca:   {max(train_tgt_lens)} kelime")
        
        print("\n[STATS] Minimum Kelime Sayısı (Train):")
        print(f"   İngilizce: {min(train_src_lens)} kelime")
        print(f"   Almanca:   {min(train_tgt_lens)} kelime")
        
        # Toplam veri
        total = len(train_data) + len(val_data) + len(test_data)
        print("\n[STATS] Toplam:")
        print(f"   {total} cümle çifti")
        print(f"   Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
        print(f"   Valid: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
        print(f"   Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")
        
        print("\n" + "=" * 70)
        print("[DONE] Multi30k dataset başarıyla yüklendi!")
        print("=" * 70)
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"\n[ERROR] Hata: {str(e)}")
        print("\n[INFO] Çözüm önerileri:")
        print("   1. İnternet bağlantınızı kontrol edin")
        print("   2. 'datasets' paketinin kurulu olduğundan emin olun: pip install datasets")
        print("   3. Manuel indirmek için: https://github.com/multi30k/dataset")
        raise


def get_vocab_stats(data, language="source"):
   
    from collections import Counter
    
  
    words = []
    idx = 0 if language == "source" else 1
    
    for item in data:
        sentence = item[idx]
        words.extend(sentence.lower().split())
    
    # İstatistikler
    word_counts = Counter(words)
    
    return {
        "total_tokens": len(words),
        "unique_tokens": len(word_counts),
        "most_common": word_counts.most_common(10),
        "vocab_size_min2": len([w for w, c in word_counts.items() if c >= 2]),
        "vocab_size_min5": len([w for w, c in word_counts.items() if c >= 5]),
    }


def main():
    
    # Config yükle
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Dataset'i indir
    train_data, val_data, test_data = download_multi30k()
    
    # Vocabulary istatistikleri
    print("\n" + "=" * 70)
    print("[INFO] Vocabulary İstatistikleri:")
    print("=" * 70)
    
    # İngilizce
    src_stats = get_vocab_stats(train_data, "source")
    print("\nSource EN (İngilizce):")
    print(f"   Toplam token: {src_stats['total_tokens']:,}")
    print(f"   Unique token: {src_stats['unique_tokens']:,}")
    print(f"   Vocab size (min_freq=2): {src_stats['vocab_size_min2']:,}")
    print(f"   Vocab size (min_freq=5): {src_stats['vocab_size_min5']:,}")
    print(f"\n   En sık 10 kelime:")
    for word, count in src_stats['most_common']:
        print(f"      {word:15s}: {count:5d}")
    
    # Almanca
    tgt_stats = get_vocab_stats(train_data, "target")
    print("\nTarget DE (Almanca):")
    print(f"   Toplam token: {tgt_stats['total_tokens']:,}")
    print(f"   Unique token: {tgt_stats['unique_tokens']:,}")
    print(f"   Vocab size (min_freq=2): {tgt_stats['vocab_size_min2']:,}")
    print(f"   Vocab size (min_freq=5): {tgt_stats['vocab_size_min5']:,}")
    print(f"\n   En sık 10 kelime:")
    for word, count in tgt_stats['most_common']:
        print(f"      {word:15s}: {count:5d}")
    
    print("\n" + "=" * 70)
    print("[DONE] Veri indirme tamamlandı!")
    print("=" * 70)
    print("\n[INFO] Veriler cache'lendi: ~/.torchtext/cache/")
    print("\n[INFO] Bir sonraki adım:")
    print(f"   python train.py --attention bahdanau")


if __name__ == "__main__":
    main()

