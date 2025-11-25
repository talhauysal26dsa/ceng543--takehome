

import os
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
   
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_imdb_dataset(data_dir='../data/raw'):
    print("=" * 60)
    print("IMDb DATASET İNDİRME")
    print("=" * 60)
    
    # Klasörü oluştur
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset URL'si
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_path / "aclImdb_v1.tar.gz"
    
    if (data_path / "aclImdb").exists():
        print("✓ Dataset zaten mevcut!")
        print(f"  Konum: {data_path / 'aclImdb'}")
        return
    
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='İndiriliyor') as t:
        urllib.request.urlretrieve(url, tar_path, reporthook=t.update_to)
    
    print("\n İndirme tamamlandı!")
    
    # Extract et
    print("\n Dosyalar extract ediliyor...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_path)
    
    print(" Extract tamamlandı!")
    
    os.remove(tar_path)
    
    # İstatistikler
    print("\n" + "=" * 60)
    print("DATASET İSTATİSTİKLERİ")
    print("=" * 60)
    
    dataset_path = data_path / "aclImdb"
    
    train_pos = len(list((dataset_path / "train" / "pos").glob("*.txt")))
    train_neg = len(list((dataset_path / "train" / "neg").glob("*.txt")))
    test_pos = len(list((dataset_path / "test" / "pos").glob("*.txt")))
    test_neg = len(list((dataset_path / "test" / "neg").glob("*.txt")))
    
    print(f"Eğitim Seti:")
    print(f"  - Pozitif: {train_pos:,} yorum")
    print(f"  - Negatif: {train_neg:,} yorum")
    print(f"  - Toplam:  {train_pos + train_neg:,} yorum")
    print(f"\nTest Seti:")
    print(f"  - Pozitif: {test_pos:,} yorum")
    print(f"  - Negatif: {test_neg:,} yorum")
    print(f"  - Toplam:  {test_pos + test_neg:,} yorum")
    print(f"\nGenel Toplam: {train_pos + train_neg + test_pos + test_neg:,} yorum")
    print("=" * 60)
    
    print("\n IMDb dataset başarıyla indirildi ve hazır!")
    print(f" Konum: {dataset_path}")
    
    # Örnek bir yorum göster
    print("\n" + "=" * 60)
    print("ÖRNEK YORUM")
    print("=" * 60)
    
    sample_file = list((dataset_path / "train" / "pos").glob("*.txt"))[0]
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print(f"\nDosya: {sample_file.name}")
    print(f"Etiket: Pozitif (Olumlu)")
    print(f"\nİçerik (ilk 300 karakter):")
    print("-" * 60)
    print(sample_text[:300] + "...")
    print("-" * 60)


if __name__ == "__main__":
    download_imdb_dataset()

