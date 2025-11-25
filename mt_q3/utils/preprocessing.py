import re
from typing import List, Tuple, Optional
from collections import Counter

import torch

class Vocabulary:
        def __init__(self, pad_token="<pad>", sos_token="<sos>", eos_token="<eos>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.itos = [pad_token, sos_token, eos_token, unk_token]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi[self.unk_token])

    def build_vocab(self, texts: List[str], min_freq: int = 2, max_size: "Optional[int]" = None):
        all_tokens = []
        for text in texts:
            all_tokens.extend(tokenize(text))
        counter = Counter(all_tokens)
        vocab_items = [(t, c) for t, c in counter.items() if c >= min_freq]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        if max_size is not None:
            vocab_items = vocab_items[:max_size]
        for token, _ in vocab_items:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        print(f"[OK] Vocab built: {len(self)} tokens (min_freq={min_freq})")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self[token] for token in tokenize(text)]
        if add_special_tokens:
            ids = [self.stoi[self.sos_token]] + ids + [self.stoi[self.eos_token]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.stoi[self.pad_token], self.stoi[self.sos_token], self.stoi[self.eos_token]}
        tokens: List[str] = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            tokens.append(self.itos[idx] if idx < len(self.itos) else self.unk_token)
        return " ".join(tokens)

class HFVocab:
        def __init__(self, tokenizer, pad_token="<pad>", sos_token="<sos>", eos_token="<eos>"):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Ensure special tokens exist in tokenizer
        special_tokens = {}
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = pad_token
        if tokenizer.cls_token is None:
            special_tokens["cls_token"] = sos_token
        if tokenizer.sep_token is None:
            special_tokens["sep_token"] = eos_token
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

        # stoi/itos compatible with base tokenizer vocab
        vocab_dict = tokenizer.get_vocab()
        self.stoi = vocab_dict
        # itos should be ordered by id; create list sized to vocab size
        self.itos = [None] * len(vocab_dict)
        for token, idx in vocab_dict.items():
            if idx < len(self.itos):
                self.itos[idx] = token
        # Fill any gaps with unk token string
        for i, tok in enumerate(self.itos):
            if tok is None:
                self.itos[i] = tokenizer.unk_token or "<unk>"

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.tokenizer.unk_token, 0))

    @property
    def pad_id(self):
        return self.stoi[self.tokenizer.pad_token]

    def encode(self, text: str, max_length: int = 50) -> List[int]:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        return encoded

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"([.!?,:;])", r" \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = torch.tensor([len(src) for src in src_batch])
    tgt_lengths = torch.tensor([len(tgt) for tgt in tgt_batch])
    max_src = src_lengths.max().item()
    max_tgt = tgt_lengths.max().item()
    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : len(src)] = src
        tgt_padded[i, : len(tgt)] = tgt
    return src_padded, tgt_padded, src_lengths, tgt_lengths

def preprocess_example(src_text: str, tgt_text: str, src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_length: int = 50):
        if isinstance(src_vocab, HFVocab) and isinstance(tgt_vocab, HFVocab):
        src_ids = src_vocab.encode(src_text, max_length=max_length)
        tgt_ids = tgt_vocab.encode(tgt_text, max_length=max_length)
    else:
        src_ids = src_vocab.encode(src_text, add_special_tokens=True)
        tgt_ids = tgt_vocab.encode(tgt_text, add_special_tokens=True)
        if len(src_ids) > max_length:
            src_ids = src_ids[: max_length - 1] + [src_vocab.stoi[src_vocab.eos_token]]
        if len(tgt_ids) > max_length:
            tgt_ids = tgt_ids[: max_length - 1] + [tgt_vocab.stoi[tgt_vocab.eos_token]]
    return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
