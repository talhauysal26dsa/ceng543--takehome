import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_hotpotqa(split="validation"):
        print(f"📥 HotpotQA {split} split'i indiriliyor...")
    print("⏱️  İlk indirme 1-2 dakika sürebilir...")
    
    try:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        print(f"✅ Başarılı! {len(ds)} örnek indirildi.")
        print(f"📁 Cache konumu: ~/.cache/huggingface/datasets/")
        
        # Örnek göster
        print("\n📋 İlk örnek:")
        example = ds[0]
        print(f"  Soru: {example['question']}")
        print(f"  Cevap: {example['answer']}")
        print(f"  Context paragrafları: {len(example['context'])}")
        print(f"  Supporting facts: {len(example['supporting_facts'])}")
        
        return ds
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("\n💡 İnternet bağlantınızı kontrol edin.")
        print("   Firewall HuggingFace'i engelliyor olabilir.")
        return None

def verify_cache():
        print("\n🔍 Cache kontrol ediliyor...")
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        print(f"✅ Cache'de {len(ds)} örnek mevcut.")
        return True
    except:
        print("❌ Cache'de veri yok veya bozuk.")
        return False

def main():
    parser = argparse.ArgumentParser(description="HotpotQA veri seti indir")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="İndirilecek split (default: validation)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Sadece cache'i kontrol et, indirme",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_cache()
    else:
        download_hotpotqa(args.split)
        print("\n✨ Hazır! Artık deneyleri çalıştırabilirsiniz:")
        print("   python -m src.run_experiments --retriever bm25 --sample-size 10")

if __name__ == "__main__":
    main()
