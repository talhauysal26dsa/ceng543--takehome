from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import yaml

from .preprocessing import Vocabulary, HFVocab, preprocess_example, collate_fn

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_length=50):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        self.preprocessed = []
        for src_text, tgt_text in data:
            self.preprocessed.append(
                preprocess_example(src_text, tgt_text, src_vocab, tgt_vocab, max_length)
            )
        print(f"[OK] Prepared {len(self.preprocessed)} examples.")

    def __len__(self):
        return len(self.preprocessed)

    def __getitem__(self, idx):
        return self.preprocessed[idx]

class TranslationDataLoader:
        def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.src_vocab = None
        self.tgt_vocab = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self):
        print("=" * 70)
        print("[INFO] Loading dataset via HuggingFace datasets")
        print("=" * 70)
        dataset_name = self.config["dataset"]["name"]
        hf_dataset = load_dataset("bentrevett/multi30k") if dataset_name == "multi30k" else load_dataset("iwslt2017", "iwslt2017-en-de")

        train_pairs = [(item["en"], item["de"]) for item in hf_dataset["train"]]
        val_pairs = [(item["en"], item["de"]) for item in hf_dataset["validation"]]
        test_pairs = [(item["en"], item["de"]) for item in hf_dataset["test"]]
        print(f" Train: {len(train_pairs)} | Valid: {len(val_pairs)} | Test: {len(test_pairs)}")

        min_freq = self.config["dataset"]["min_freq"]
        max_length = self.config["dataset"]["max_length"]
        limit_train = self.config["training"].get("limit_train")
        limit_val = self.config["training"].get("limit_val")

        embed_type = self.config.get("embeddings", {}).get("type", "static")
        if embed_type == "transformer":
            from transformers import AutoTokenizer

            model_name = self.config["embeddings"]["transformer"]["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": self.config["dataset"]["pad_token"]})
            self.src_vocab = HFVocab(tokenizer)
            self.tgt_vocab = HFVocab(tokenizer)
            print(f"\n[STEP] Using transformer tokenizer vocab: {len(self.src_vocab)} tokens")
        else:
            self.src_vocab = Vocabulary(
                pad_token=self.config["dataset"]["pad_token"],
                sos_token=self.config["dataset"]["sos_token"],
                eos_token=self.config["dataset"]["eos_token"],
                unk_token=self.config["dataset"]["unk_token"],
            )
            self.tgt_vocab = Vocabulary(
                pad_token=self.config["dataset"]["pad_token"],
                sos_token=self.config["dataset"]["sos_token"],
                eos_token=self.config["dataset"]["eos_token"],
                unk_token=self.config["dataset"]["unk_token"],
            )

            print("\n[STEP] Building source vocab (EN)")
            self.src_vocab.build_vocab([s for s, _ in train_pairs], min_freq=min_freq)
            print("[STEP] Building target vocab (DE)")
            self.tgt_vocab.build_vocab([t for _, t in train_pairs], min_freq=min_freq)

        if limit_train:
            train_pairs = train_pairs[:limit_train]
        if limit_val:
            val_pairs = val_pairs[:limit_val]

        print("\n[STEP] Materializing datasets")
        train_ds = TranslationDataset(train_pairs, self.src_vocab, self.tgt_vocab, max_length)
        val_ds = TranslationDataset(val_pairs, self.src_vocab, self.tgt_vocab, max_length)
        test_ds = TranslationDataset(test_pairs, self.src_vocab, self.tgt_vocab, max_length)

        pad_idx = self.src_vocab.pad_id if isinstance(self.src_vocab, HFVocab) else self.src_vocab.stoi[self.src_vocab.pad_token]
        def collate_wrapper(batch):
            return collate_fn(batch, pad_idx)

        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["hardware"]["num_workers"]
        pin_memory = self.config["hardware"]["pin_memory"]

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_wrapper)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_wrapper)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_wrapper)

        print(f"[OK] Loaders -> Train batches: {len(self.train_loader)} | Val: {len(self.val_loader)} | Test: {len(self.test_loader)}")
        return self

    def get_vocabs(self):
        return self.src_vocab, self.tgt_vocab

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
