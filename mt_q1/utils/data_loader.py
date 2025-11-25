

import os
from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing import TextPreprocessor


class IMDbDataset(Dataset):

    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 max_samples: int = None,
                 preprocessor: TextPreprocessor = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = preprocessor or TextPreprocessor()
        
    
        self.texts, self.labels = self._load_data(max_samples)
        
        print(f"✓ {split.upper()} seti yüklendi: {len(self.texts):,} örnek")
        
    def _load_data(self, max_samples: int = None) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        
        split_dir = self.data_dir / self.split
      
        pos_dir = split_dir / 'pos'
        for file_path in pos_dir.glob('*.txt'):
            if max_samples and len(texts) >= max_samples // 2:
                break
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
                labels.append(1)  # Positive
     
        neg_dir = split_dir / 'neg'
        for file_path in neg_dir.glob('*.txt'):
            if max_samples and len(texts) >= max_samples:
                break
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
                labels.append(0)  # Negative
        
        # Shuffle et
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        return text, label
    
    def get_statistics(self) -> Dict:
        num_positive = sum(self.labels)
        num_negative = len(self.labels) - num_positive
        
        # Metin uzunlukları
        text_lengths = [len(text.split()) for text in self.texts]
        avg_length = sum(text_lengths) / len(text_lengths)
        max_length = max(text_lengths)
        min_length = min(text_lengths)
        
        return {
            'total': len(self.texts),
            'positive': num_positive,
            'negative': num_negative,
            'avg_length': avg_length,
            'max_length': max_length,
            'min_length': min_length
        }


class IMDbDataLoader:
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 max_length: int = 256,
                 max_samples: int = None,
                 num_workers: int = 0):
    
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.num_workers = num_workers
        
        self.preprocessor = TextPreprocessor(max_length=max_length)
        
        self.train_dataset = IMDbDataset(
            data_dir=data_dir,
            split='train',
            max_samples=max_samples,
            preprocessor=self.preprocessor
        )
        
        self.test_dataset = IMDbDataset(
            data_dir=data_dir,
            split='test',
            max_samples=max_samples,
            preprocessor=self.preprocessor
        )
        

        self._print_statistics()
    
    def _print_statistics(self):
    
        print("\n" + "=" * 60)
        print("DATASET İSTATİSTİKLERİ")
        print("=" * 60)
        
        train_stats = self.train_dataset.get_statistics()
        test_stats = self.test_dataset.get_statistics()
        
        print(f"\nTrain Set:")
        print(f"  Total:    {train_stats['total']:,} örnekler")
        print(f"  Positive: {train_stats['positive']:,} ({train_stats['positive']/train_stats['total']*100:.1f}%)")
        print(f"  Negative: {train_stats['negative']:,} ({train_stats['negative']/train_stats['total']*100:.1f}%)")
        print(f"  Ortalama uzunluk: {train_stats['avg_length']:.1f} kelime")
        print(f"  Max uzunluk: {train_stats['max_length']} kelime")
        
        print(f"\nTest Set:")
        print(f"  Total:    {test_stats['total']:,} örnekler")
        print(f"  Positive: {test_stats['positive']:,} ({test_stats['positive']/test_stats['total']*100:.1f}%)")
        print(f"  Negative: {test_stats['negative']:,} ({test_stats['negative']/test_stats['total']*100:.1f}%)")
        print(f"  Ortalama uzunluk: {test_stats['avg_length']:.1f} kelime")
        print(f"  Max uzunluk: {test_stats['max_length']} kelime")
        
        print("=" * 60)
    
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
    
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
    
    def get_test_loader(self) -> DataLoader:
    
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_sample_batch(self, split: str = 'train', n: int = 5):
       
        dataset = self.train_dataset if split == 'train' else self.test_dataset
        
        print(f"\n{'=' * 60}")
        print(f"ÖRNEK {split.upper()} BATCH ({n} örnek)")
        print("=" * 60)
        
        for i in range(min(n, len(dataset))):
            text, label = dataset[i]
            label_str = "POS" if label == 1 else "NEG"
            
            # İlk 100 karakter
            text_preview = text[:100] + "..." if len(text) > 100 else text
            
            print(f"\n{i+1}. [{label_str}] {text_preview}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Test
    print("Testing IMDbDataLoader...")
    
    loader = IMDbDataLoader(
        data_dir='../data/raw/aclImdb',
        batch_size=4,
        max_samples=100  # Sadece 100 örnek
    )
    
    # Örnek batch göster
    loader.get_sample_batch(split='train', n=3)
    
    # DataLoader test
    train_loader = loader.get_train_loader()
    batch_texts, batch_labels = next(iter(train_loader))
    
    print(f"\nBatch shape:")
    print(f"  Texts: {len(batch_texts)} örnekler")
    print(f"  Labels: {batch_labels.shape if isinstance(batch_labels, torch.Tensor) else len(batch_labels)}")

