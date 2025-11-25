

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pathlib import Path
import yaml

from .preprocessing import Vocabulary, preprocess_example, collate_fn


class TranslationDataset(Dataset):
    
    
    def __init__(self, data, src_vocab, tgt_vocab, max_length=50):
        
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Preprocessing yap (tüm veriyi once hazırla)
        print(f"   Preprocessing {len(data)} examples...")
        self.preprocessed_data = []
        
        for src_text, tgt_text in data:
            src_tensor, tgt_tensor = preprocess_example(
                src_text, tgt_text,
                src_vocab, tgt_vocab,
                max_length
            )
            self.preprocessed_data.append((src_tensor, tgt_tensor))
        
        print(f"   [OK] {len(self.preprocessed_data)} examples hazır")
    
    def __len__(self):
        """Dataset boyutu."""
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx):
        
        return self.preprocessed_data[idx]


class TranslationDataLoader:
    
    
    def __init__(self, config_path="config.yaml"):
       
        # Config yükle
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.src_vocab = None
        self.tgt_vocab = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def load_data(self):
       
        print("=" * 70)
        print("[INFO] Data Loading Pipeline Başlatılıyor...")
        print("=" * 70)
        
        # 1. Dataset'i yükle
        print("\n[INFO] Hugging Face 'datasets' arayüzü kullanılıyor...")
        hf_dataset = load_dataset("bentrevett/multi30k")

        train_data = [(item["en"], item["de"]) for item in hf_dataset["train"]]
        val_data = [(item["en"], item["de"]) for item in hf_dataset["validation"]]
        test_data = [(item["en"], item["de"]) for item in hf_dataset["test"]]

        print(f"   [OK] Train: {len(train_data)} examples")
        print(f"   [OK] Valid: {len(val_data)} examples")
        print(f"   [OK] Test:  {len(test_data)} examples")
        
        # 2. Vocabulary oluştur
        print("\n[STEP 2] Vocabulary oluşturuluyor...")
        
        min_freq = self.config['dataset']['min_freq']
        
        # Source vocabulary
        print(f"\n   Source (English) vocabulary:")
        self.src_vocab = Vocabulary(
            pad_token=self.config['dataset']['pad_token'],
            sos_token=self.config['dataset']['sos_token'],
            eos_token=self.config['dataset']['eos_token'],
            unk_token=self.config['dataset']['unk_token']
        )
        src_texts = [src for src, _ in train_data]
        self.src_vocab.build_vocab(src_texts, min_freq=min_freq)
        
        # Target vocabulary 
        print(f"\n   Target (German) vocabulary:")
        self.tgt_vocab = Vocabulary(
            pad_token=self.config['dataset']['pad_token'],
            sos_token=self.config['dataset']['sos_token'],
            eos_token=self.config['dataset']['eos_token'],
            unk_token=self.config['dataset']['unk_token']
        )
        tgt_texts = [tgt for _, tgt in train_data]
        self.tgt_vocab.build_vocab(tgt_texts, min_freq=min_freq)
        
        # 3. Dataset'leri oluştur
        print("\n[STEP 3] PyTorch Dataset'leri oluşturuluyor...")
        
        max_length = self.config['dataset']['max_length']
        
        print("\n   Training set:")
        train_dataset = TranslationDataset(
            train_data, self.src_vocab, self.tgt_vocab, max_length
        )
        
        print("\n   Validation set:")
        val_dataset = TranslationDataset(
            val_data, self.src_vocab, self.tgt_vocab, max_length
        )
        
        print("\n   Test set:")
        test_dataset = TranslationDataset(
            test_data, self.src_vocab, self.tgt_vocab, max_length
        )
        
        # 4. DataLoader'ları oluştur
        print("\n[STEP 4] DataLoader'lar oluşturuluyor...")
        
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware']['num_workers']
        pin_memory = self.config['hardware']['pin_memory']
        
        pad_idx = self.src_vocab.stoi[self.src_vocab.pad_token]
        
        # Collate function (padding için)
        def collate_wrapper(batch):
            return collate_fn(batch, pad_idx)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Training'de shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_wrapper
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_wrapper
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Test'te shuffle yok
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_wrapper
        )
        
        print(f"   [OK] Train: {len(self.train_loader)} batches")
        print(f"   [OK] Valid: {len(self.val_loader)} batches")
        print(f"   [OK] Test:  {len(self.test_loader)} batches")
        
        # 5. Özet
        print("\n" + "=" * 70)
        print("[SUMMARY] Data Loading Özeti:")
        print("=" * 70)
        print("\n[VOCAB] Vocabulary Boyutları:")
        print(f"   Source (EN): {len(self.src_vocab)} tokens")
        print(f"   Target (DE): {len(self.tgt_vocab)} tokens")
        
        print("\n[DATASET] Dataset Boyutları:")
        print(f"   Train: {len(train_dataset)} examples -> {len(self.train_loader)} batches")
        print(f"   Valid: {len(val_dataset)} examples -> {len(self.val_loader)} batches")
        print(f"   Test:  {len(test_dataset)} examples -> {len(self.test_loader)} batches")
        
        print("\n[SETTINGS] Hyperparameters:")
        print(f"   Batch size: {batch_size}")
        print(f"   Max length: {max_length}")
        print(f"   Min freq: {min_freq}")
        
        print("\n" + "=" * 70)
        print("[DONE] Data loading tamamlandı!")
        print("=" * 70)
        
        return self
    
    def get_train_loader(self):
        """Training DataLoader'ı al."""
        if self.train_loader is None:
            raise ValueError("load_data() henüz çağrılmadı!")
        return self.train_loader
    
    def get_val_loader(self):
        """Validation DataLoader'ı al."""
        if self.val_loader is None:
            raise ValueError("load_data() henüz çağrılmadı!")
        return self.val_loader
    
    def get_test_loader(self):
        """Test DataLoader'ı al."""
        if self.test_loader is None:
            raise ValueError("load_data() henüz çağrılmadı!")
        return self.test_loader
    
    def get_vocabs(self):
        """Vocabulary'leri al."""
        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("load_data() henüz çağrılmadı!")
        return self.src_vocab, self.tgt_vocab


def test_data_loader():
    """Test: Data loading pipeline'ı test et."""
    
    print("=" * 70)
    print("[TEST] Data Loader")
    print("=" * 70)
    
    # Config path
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    # DataLoader oluştur
    data_loader = TranslationDataLoader(config_path)
    data_loader.load_data()
    
    # Vocabulary'leri al
    src_vocab, tgt_vocab = data_loader.get_vocabs()
    
    # Bir batch al
    print("\n" + "=" * 70)
    print("[BATCH] Örnek Batch:")
    print("=" * 70)
    
    train_loader = data_loader.get_train_loader()
    src_batch, tgt_batch, src_lengths, tgt_lengths = next(iter(train_loader))
    
    print(f"\nBatch shape:")
    print(f"   src_batch: {src_batch.shape}")  # [batch_size, max_src_len]
    print(f"   tgt_batch: {tgt_batch.shape}")  # [batch_size, max_tgt_len]
    print(f"   src_lengths: {src_lengths.shape}")  # [batch_size]
    print(f"   tgt_lengths: {tgt_lengths.shape}")  # [batch_size]
    
    # İlk örneği decode et
    print("\n[NOTE] İlk örnek:")
    src_ids = src_batch[0].tolist()
    tgt_ids = tgt_batch[0].tolist()
    
    src_text = src_vocab.decode(src_ids)
    tgt_text = tgt_vocab.decode(tgt_ids)
    
    print(f"   Source EN (len={src_lengths[0]}): {src_text}")
    print(f"   Target DE (len={tgt_lengths[0]}): {tgt_text}")
    
    print(f"\n   Source IDs (ilk 10): {src_ids[:10]}")
    print(f"   Target IDs (ilk 10): {tgt_ids[:10]}")
    
    print("\n" + "=" * 70)
    print("[DONE] Data loader test tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    test_data_loader()

