"""
GloVe embeddings indirme scripti.

GloVe (Global Vectors for Word Representation):
- Stanford tarafından geliştirilmiş
- 6 milyar token üzerinde eğitilmiş
- 300 boyutlu vektörler
- 400,000 kelime içerir
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_glove_embeddings(embeddings_dir='../embeddings/glove'):
    
    
    # Klasör oluştur
    emb_path = Path(embeddings_dir)
    emb_path.mkdir(parents=True, exist_ok=True)
    
    # Target file
    target_file = emb_path / "glove.6B.300d.txt"
    
    # Eğer zaten varsa skip et
    if target_file.exists():
        print(" GloVe embeddings zaten mevcut!")
        print(f"  Konum: {target_file}")
        
        # İstatistikler
        print("\n İstatistikler:")
        with open(target_file, 'r', encoding='utf-8') as f:
            num_words = sum(1 for _ in f)
        print(f"  - Kelime sayısı: {num_words:,}")
        print(f"  - Vektör boyutu: 300")
        print(f"  - Dosya boyutu: {target_file.stat().st_size / (1024**2):.1f} MB")
        return
    
    # URL
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = emb_path / "glove.6B.zip"
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='İndiriliyor') as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
        
        print("\n İndirme tamamlandı!")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('glove.6B.300d.txt', emb_path)
        
        print(" Extract tamamlandı!")
        
        os.remove(zip_path)
        
        # İstatistikler
        print("\n" + "=" * 60)
        print("GLOVE İSTATİSTİKLERİ")
        print("=" * 60)
        
        with open(target_file, 'r', encoding='utf-8') as f:
            num_words = sum(1 for _ in f)
        
        print(f"Kelime sayısı: {num_words:,}")
        print(f"Vektör boyutu: 300")
        print(f"Dosya boyutu: {target_file.stat().st_size / (1024**2):.1f} MB")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("ÖRNEK KELİMELER")
        print("=" * 60)
        
        print("\nİlk 5 kelime ve vektörlerinin başlangıcı:")
        with open(target_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                parts = line.split()
                word = parts[0]
                vec_start = ' '.join(parts[1:6])  # İlk 5 boyut
                print(f"  {word:12} → [{vec_start} ...]")
        
        print("\n GloVe embeddings başarıyla indirildi ve hazır!")
        
    except Exception as e:
        print(f"\ Hata oluştu: {e}")


if __name__ == "__main__":
    download_glove_embeddings()

