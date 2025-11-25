

import torch
from collections import Counter
from typing import List, Tuple, Dict
import re


class Vocabulary:
    
    
    def __init__(self, pad_token="<pad>", sos_token="<sos>", 
                 eos_token="<eos>", unk_token="<unk>"):
        
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
        # Special token'ları başa ekle
        self.itos = [pad_token, sos_token, eos_token, unk_token]  
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}  
        
    def __len__(self):
      
        return len(self.itos)
    
    def __getitem__(self, token):
        
        return self.stoi.get(token, self.stoi[self.unk_token])
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_size: int = None):
        
        # Tüm kelimeleri topla
        all_tokens = []
        for text in texts:
            tokens = tokenize(text)
            all_tokens.extend(tokens)
        
        # Frekans say
        counter = Counter(all_tokens)
        
        # min_freq'den fazla olanları al
        vocab_items = [(token, count) for token, count in counter.items() 
                       if count >= min_freq]
        
        
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        
        # max_size varsa kırp
        if max_size is not None:
            vocab_items = vocab_items[:max_size]
        
        # Vocabulary'ye ekle
        for token, count in vocab_items:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        
        print(f"[OK] Vocabulary oluşturuldu: {len(self)} kelime (min_freq={min_freq})")
        print(f"  - Special tokens: {len([self.pad_token, self.sos_token, self.eos_token, self.unk_token])}")
        print(f"  - Regular tokens: {len(self) - 4}")
        print(f"  - Toplam tokens görüldü: {len(all_tokens)}")
        print(f"  - Unique tokens: {len(counter)}")
        print(f"  - Filtrelenenler (freq<{min_freq}): {len(counter) - (len(self) - 4)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        
        tokens = tokenize(text)
        ids = [self[token] for token in tokens]
        
        if add_special_tokens:
            ids = [self.stoi[self.sos_token]] + ids + [self.stoi[self.eos_token]]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        
        special_ids = {self.stoi[self.pad_token], 
                      self.stoi[self.sos_token], 
                      self.stoi[self.eos_token]}
        
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            if idx < len(self.itos):
                tokens.append(self.itos[idx])
            else:
                tokens.append(self.unk_token)
        
        return " ".join(tokens)


def tokenize(text: str) -> List[str]:
  
    # Küçük harfe çevir
    text = text.lower()
    
    
    text = re.sub(r"([.!?,:;])", r" \1", text)
    
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    
    # Boşluklara göre böl
    tokens = text.split()
    
    return tokens


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], 
               pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
 
    # Batch'i ayır
    src_batch, tgt_batch = zip(*batch)
    
    # Uzunlukları bul
    src_lengths = torch.tensor([len(src) for src in src_batch])
    tgt_lengths = torch.tensor([len(tgt) for tgt in tgt_batch])
    
    # Maksimum uzunlukları bul
    max_src_len = src_lengths.max().item()
    max_tgt_len = tgt_lengths.max().item()
    
    # Padding yap
    src_padded = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths


def preprocess_example(src_text: str, tgt_text: str, 
                       src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                       max_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
     
    # Encode et
    src_ids = src_vocab.encode(src_text, add_special_tokens=True)
    tgt_ids = tgt_vocab.encode(tgt_text, add_special_tokens=True)
    
    # Max length kontrolü
    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length-1] + [src_vocab.stoi[src_vocab.eos_token]]
    
    if len(tgt_ids) > max_length:
        tgt_ids = tgt_ids[:max_length-1] + [tgt_vocab.stoi[tgt_vocab.eos_token]]
    
    # Tensor'e çevir
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    
    return src_tensor, tgt_tensor


if __name__ == "__main__":
    
    print("=" * 70)
    print(" Preprocessing Test")
    print("=" * 70)
    
    # Test data
    texts = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love"
    ]
    
    print("\n Tokenization:")
    for text in texts:
        tokens = tokenize(text)
        print(f"   '{text}' → {tokens}")
    
    print("\n  Vocabulary Building:")
    vocab = Vocabulary()
    vocab.build_vocab(texts, min_freq=1)
    
    print(f"\n   Vocabulary içeriği:")
    for i, token in enumerate(vocab.itos):
        print(f"      {i}: {token}")
    
    print("\n  Encoding:")
    test_text = "hello world peace"
    encoded = vocab.encode(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Tokens: {[vocab.itos[i] for i in encoded]}")
    
    print("\n  Decoding:")
    decoded = vocab.decode(encoded)
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")
    
    print("\n Unknown Token Test:")
    unknown_text = "hello xyz"  # "xyz" vocabulary'de yok
    encoded = vocab.encode(unknown_text)
    print(f"   Text: '{unknown_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Tokens: {[vocab.itos[i] for i in encoded]}")
    
    print("\n" + "=" * 70)
    print(" Preprocessing test tamamlandı!")
    print("=" * 70)

