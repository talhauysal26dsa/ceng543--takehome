import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from utils.data_loader import TranslationDataLoader
from models.seq2seq import create_seq2seq_model
from models.transformer import TransformerNMT

def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch_seq2seq(model, dataloader, optimizer, criterion, clip, device, teacher_forcing):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc="Train Seq2Seq", leave=False)
    for src, tgt, src_lengths, _ in pbar:
        src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
        optimizer.zero_grad()
        output, _ = model(src, src_lengths, tgt, teacher_forcing_ratio=teacher_forcing)
        output = output[:, 1:, :].contiguous()
        tgt_shift = tgt[:, 1:].contiguous()
        loss = criterion(output.view(-1, output.shape[-1]), tgt_shift.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return epoch_loss / len(dataloader)

def eval_seq2seq(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval Seq2Seq", leave=False)
        for src, tgt, src_lengths, _ in pbar:
            src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
            output, _ = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
            output = output[:, 1:, :].contiguous()
            tgt_shift = tgt[:, 1:].contiguous()
            loss = criterion(output.view(-1, output.shape[-1]), tgt_shift.view(-1))
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    return epoch_loss / len(dataloader)

def train_epoch_transformer(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc="Train Transformer", leave=False)
    for src, tgt, _, _ in pbar:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return epoch_loss / len(dataloader)

def eval_transformer(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Eval Transformer", leave=False)
        for src, tgt, _, _ in pbar:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    return epoch_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, choices=["seq2seq", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    set_seed(config["seeds"]["torch_seed"])
    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    data = TranslationDataLoader(args.config).load_data()
    train_loader, val_loader = data.get_train_loader(), data.get_val_loader()
    src_vocab, tgt_vocab = data.get_vocabs()

    save_dir = Path(config["logging"]["save_dir"]) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    pad_idx = tgt_vocab.pad_id if hasattr(tgt_vocab, "pad_id") else tgt_vocab.stoi[tgt_vocab.pad_token]
    src_pad_idx = src_vocab.pad_id if hasattr(src_vocab, "pad_id") else src_vocab.stoi[src_vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if args.model == "seq2seq":
        model = create_seq2seq_model(len(src_vocab), len(tgt_vocab), config, device, src_pad_idx=src_pad_idx, tgt_pad_idx=pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        train_epoch_fn = lambda dl: train_epoch_seq2seq(model, dl, optimizer, criterion, config["training"]["gradient_clip"], device, config["seq2seq"]["teacher_forcing"])
        eval_epoch_fn = lambda dl: eval_seq2seq(model, dl, criterion, device)
    else:
        model = TransformerNMT(len(src_vocab), len(tgt_vocab), config, pad_idx=pad_idx).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        train_epoch_fn = lambda dl: train_epoch_transformer(model, dl, optimizer, criterion, device)
        eval_epoch_fn = lambda dl: eval_transformer(model, dl, criterion, device)

    print(f"[INFO] Model parameters: {count_parameters(model):,}")

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "epoch_time_sec": [], "max_gpu_mem_mb": []}

    use_cuda = device.type == "cuda"
    save_last = config.get("logging", {}).get("save_last", True)
    save_best = config.get("logging", {}).get("save_best", True)

    patience = config["training"].get("early_stopping_patience", 5)
    use_early_stop = config["training"].get("early_stopping", False)
    epochs_no_improve = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        train_loss = train_epoch_fn(train_loader)
        val_loss = eval_epoch_fn(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.time() - start
        history["epoch_time_sec"].append(elapsed)
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_cuda else None
        history["max_gpu_mem_mb"].append(max_mem)
        if save_last:
            torch.save(model.state_dict(), save_dir / f"last_{epoch}.pt", _use_new_zipfile_serialization=False)
        if val_loss < best_val and save_best:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt", _use_new_zipfile_serialization=False)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        mem_str = f" | gpu_mem: {max_mem:.1f} MB" if max_mem is not None else ""
        print(f"[EPOCH {epoch}] train: {train_loss:.4f} | val: {val_loss:.4f} | time: {elapsed:.1f}s{mem_str}")

        if use_early_stop and epochs_no_improve >= patience:
            print(f"[EARLY STOP] No val improvement for {patience} epochs. Best val: {best_val:.4f}")
            break
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[DONE] Training completed. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
