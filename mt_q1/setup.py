
import os
import sys
from pathlib import Path


def print_header(text):
    """Başlık yazdır."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num, total_steps, text):
    
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)


def check_python_version():

    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(" Python 3.8 veya üzeri gerekli!")
        print("   Lütfen Python'ı güncelleyin: https://www.python.org/downloads/")
        sys.exit(1)
    
    print(" Python version uygun!")


def check_packages():
    
    print_header("PACKAGE CHECK")
    
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'nltk',
        'sklearn',
        'matplotlib',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package:20} ... OK")
        except ImportError:
            print(f" {package:20} ... MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n {len(missing_packages)} paket eksik!")
        print("   Yüklemek için: pip install -r requirements.txt")
        
        response = input("\nŞimdi yüklemek ister misiniz? (y/N): ")
        if response.lower() == 'y':
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("Lütfen manuel olarak yükleyin ve tekrar çalıştırın.")
            sys.exit(1)
    else:
        print("\n Tüm paketler yüklü!")


def create_directories():
    
    print_header("CREATING DIRECTORIES")
    
    dirs = [
        "data/raw",
        "data/processed",
        "embeddings/glove",
        "embeddings/cache",
        "experiments/lstm_glove",
        "experiments/lstm_bert",
        "experiments/gru_glove",
        "experiments/gru_bert",
        "experiments/plots",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f" Created: {dir_path}")
    
    print("\n Tüm klasörler oluşturuldu!")


def download_imdb():
    
    print_header("DOWNLOADING IMDB DATASET")
    
    
    if Path("data/raw/aclImdb").exists():
        print(" IMDb dataset zaten mevcut!")
        return
    
    print("  IMDb dataset indiriliyor (~84 MB)...")
    print("   Bu işlem birkaç dakika sürebilir.\n")
    
    from utils.download_data import download_imdb_dataset
    
    try:
        download_imdb_dataset(data_dir='data/raw')
        print("\n  IMDb dataset başarıyla indirildi!")
    except Exception as e:
        print(f"\n  Hata: {e}")
        print("   Lütfen manuel olarak indirin:")
        print("   python utils/download_data.py")


def download_glove():
   
    print_header("DOWNLOADING GLOVE EMBEDDINGS")
    
    # Check if already exists
    glove_path = Path("embeddings/glove/glove.6B.300d.txt")
    if glove_path.exists():
        print(" GloVe embeddings zaten mevcut!")
        return
    
    print("  GloVe embeddings indiriliyor (~862 MB)...")
    print("   Bu işlem uzun sürebilir (internet hızınıza bağlı).")
    print("   Lütfen bekleyin...\n")
    
    from utils.download_glove import download_glove_embeddings
    
    try:
        download_glove_embeddings(embeddings_dir='embeddings/glove')
        print("\n GloVe embeddings başarıyla indirildi!")
    except Exception as e:
        print(f"\n  Hata: {e}")
        print("   Lütfen manuel olarak indirin:")
        print("   python utils/download_glove.py")


def run_tests():
   
    print_header("RUNNING TESTS")
    
    tests = [
        ("Preprocessing", "utils/preprocessing.py"),
        ("Embedding Loader", "utils/embedding_loader.py"),
        ("LSTM Model", "models/lstm_model.py"),
        ("GRU Model", "models/gru_model.py"),
    ]
    
    print("Basit testler çalıştırılıyor...\n")
    
    for test_name, test_file in tests:
        print(f"Testing {test_name}...")
        if Path(test_file).exists():
            print(f"   {test_file} mevcut")
        else:
            print(f"   {test_file} bulunamadı!")
    
    print("\n  Detaylı test için:")
    print("   python utils/preprocessing.py")
    print("   python models/lstm_model.py")


def print_next_steps():
    
    print_header("SETUP COMPLETE! 🎉")
    
    print(" Proje başarıyla kuruldu!\n")
    
    print("📚 Sonraki Adımlar:")
    print("\n  Hızlı Test (Küçük Dataset):")
    print("   python train.py --model lstm --embedding glove --max_samples 1000 --epochs 3")
    
    print("\n  Tek Model Eğit:")
    print("   python train.py --model lstm --embedding glove --epochs 10")
    
    print("\n  Tüm Modelleri Eğit:")
    print("   python train.py --model all --embedding all --epochs 10")
    
    print("\n Sonuçları Değerlendir:")
    print("   python evaluate.py --compare")
    
    print("\n Grafikleri Oluştur:")
    print("   python visualize.py")
    
    print("\n Daha fazla bilgi için:")
    print("   - README.md")
    print("   - PROJE_ACIKLAMA.md")
    print("   - HIZLI_BASLANGIC.md")
    
    print("\n" + "=" * 70)


def main():

    print("\n" + "=" * 70)
    print("  SENTIMENT ANALYSIS PROJECT - AUTOMATED SETUP")
    print("=" * 70)
    
    total_steps = 6
    
    print_step(1, total_steps, "Python Version Check")
    check_python_version()
    
    print_step(2, total_steps, "Package Installation Check")
    check_packages()
    
    print_step(3, total_steps, "Creating Directory Structure")
    create_directories()
    
    print_step(4, total_steps, "Downloading IMDb Dataset")
    response = input("\nIMDb dataset'ini indirmek ister misiniz? (~84 MB) (Y/n): ")
    if response.lower() != 'n':
        download_imdb()
    else:
        print(" Atlandı. Manuel indirme: python utils/download_data.py")
    
    
    print_step(5, total_steps, "Downloading GloVe Embeddings")
    response = input("\nGloVe embeddings'i indirmek ister misiniz? (~862 MB) (Y/n): ")
    if response.lower() != 'n':
        download_glove()
    else:
        print(" Atlandı. Manuel indirme: python utils/download_glove.py")
    
    # Step 6: Tests
    print_step(6, total_steps, "Running Basic Tests")
    run_tests()
    
    # Next steps
    print_next_steps()


if __name__ == "__main__":
    main()

