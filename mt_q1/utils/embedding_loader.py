

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm


class EmbeddingLoader:
    
    def __init__(self, embedding_path: str, embedding_dim: int = 300):
        self.embedding_path = Path(embedding_path)
        self.embedding_dim = embedding_dim
        self.embeddings = None
        
        # Embedding'leri yükle
        if self.embedding_path.exists():
            self._load_embeddings()
        else:
            print(f"   Embedding dosyası bulunamadı: {embedding_path}")
            print(f"   Lütfen önce embedding'leri indirin:")
            print(f"   python utils/download_glove.py")
    
    def _load_embeddings(self):
        print(f"\n Embedding yükleniyor: {self.embedding_path.name}")
        print(f"   Boyut: {self.embedding_dim}")
        
        self.embeddings = {}
        
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
           
            first_line = f.readline()
            parts = first_line.split()
            

            if len(parts) == 2 and parts[0].isdigit():
                
                pass
            else:
               
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.embeddings[word] = vector
            
            for line in tqdm(f, desc="Yükleniyor", unit=" kelime"):
                parts = line.split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self.embeddings[word] = vector
        
        print(f" {len(self.embeddings):,} kelime yüklendi")
        
        sample_words = list(self.embeddings.keys())[:5]
        print(f"\nÖrnek kelimeler: {', '.join(sample_words)}")
    
    def get_embedding(self, word: str) -> np.ndarray:
        
        if self.embeddings is None:
            return None
        return self.embeddings.get(word.lower())
    
    def create_embedding_matrix(self, 
                               word_to_idx: Dict[str, int],
                               trainable_oov: bool = True) -> torch.Tensor:
        print(f"\n Embedding matrix oluşturuluyor...")
        print(f"   Vocabulary boyutu: {len(word_to_idx):,}")
        print(f"   Embedding boyutu: {self.embedding_dim}")
        
        vocab_size = len(word_to_idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        
        found_count = 0
        oov_count = 0
        
        for word, idx in tqdm(word_to_idx.items(), desc="Mapping"):
            if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                embedding_matrix[idx] = np.zeros(self.embedding_dim)
                continue
            
            # GloVe'da ara
            embedding = self.get_embedding(word)
            
            if embedding is not None:
                embedding_matrix[idx] = embedding
                found_count += 1
            else:
                if trainable_oov:
                    # Xavier/Glorot initialization
                    limit = np.sqrt(6.0 / self.embedding_dim)
                    embedding_matrix[idx] = np.random.uniform(-limit, limit, self.embedding_dim)
                else:
                    embedding_matrix[idx] = np.zeros(self.embedding_dim)
                oov_count += 1
        
        
        coverage = (found_count / vocab_size) * 100
        print(f"\n✓ Embedding matrix hazır!")
        print(f"   Found: {found_count:,} kelime ({coverage:.1f}%)")
        print(f"   OOV:   {oov_count:,} kelime ({100-coverage:.1f}%)")
        
        # Tensor'a çevir
        embedding_tensor = torch.from_numpy(embedding_matrix)
        
        return embedding_tensor
    
    def find_similar_words(self, word: str, topk: int = 5):
        
        if self.embeddings is None:
            print(" Embedding'ler yüklenmemiş!")
            return
        
        word = word.lower()
        if word not in self.embeddings:
            print(f"'{word}' kelimesi embedding'lerde bulunamadı!")
            return
        
        target_vec = self.embeddings[word]
        
        similarities = {}
        for w, vec in self.embeddings.items():
            if w == word:
                continue
            sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec))
            similarities[w] = sim
        
        # En yüksek skorları bul
        top_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:topk]
        
        print(f"\n'{word}' kelimesine en benzer {topk} kelime:")
        for i, (w, score) in enumerate(top_words, 1):
            print(f"  {i}. {w:15} (similarity: {score:.4f})")


if __name__ == "__main__":
    print("Testing EmbeddingLoader...")
    
    script_dir = Path(__file__).parent.parent  
    embedding_path = script_dir / 'embeddings' / 'glove' / 'glove.6B.300d.txt'
    
    # GloVe yükle
    loader = EmbeddingLoader(
        embedding_path=str(embedding_path),
        embedding_dim=300
    )
    
    if loader.embeddings is None:
        print("\n  Embeddings yüklenemedi, test atlanıyor.")
        print(f"   Lütfen dosyanın mevcut olduğundan emin olun: {embedding_path}")
        exit(1)
    
    # Örnek kelimeler
    test_words = ['good', 'bad', 'movie', 'film', 'actor']
    
    print("\n" + "=" * 60)
    print("ÖRNEK KELİME VEKTÖRLERİ")
    print("=" * 60)
    
    for word in test_words:
        vec = loader.get_embedding(word)
        if vec is not None:
            print(f"\n{word:10} → [{vec[:5]}...]")
            print(f"           Boyut: {len(vec)}")
        else:
            print(f"\n{word:10} → Bulunamadı")
    
    # Benzer kelimeler
    print("\n" + "=" * 60)
    print("BENZERLİK TESTİ")
    print("=" * 60)
    
    loader.find_similar_words('good', topk=5)
    loader.find_similar_words('movie', topk=5)

