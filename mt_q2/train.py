

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import json
import os

from models.seq2seq import create_seq2seq_model
from utils.data_loader import TranslationDataLoader


def set_seed(seed):
   
    print(f"\n Setting random seeds: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("   [OK] Python random seed set")
    print("   [OK] NumPy random seed set")
    print("   [OK] PyTorch random seed set")
    print("   [OK] CUDA random seed set")
    print("   [OK] cuDNN deterministic mode enabled")


def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    
    model.train()
    epoch_loss = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(pbar):
        # Move to device
        src = src.to(device)
        tgt = tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Output: [batch_size, tgt_len, vocab_size]
        output, _ = model(src, src_lengths, tgt)
        
        # Loss hesapla
        
        output = output[:, 1:, :].contiguous()  # [batch, tgt_len-1, vocab]
        tgt = tgt[:, 1:].contiguous()            # [batch, tgt_len-1]
        
        # Reshape for loss
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)  # [batch * (tgt_len-1), vocab]
        tgt = tgt.view(-1)                     # [batch * (tgt_len-1)]
        
        # Loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (exploding gradient'ı önler)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
   
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for src, tgt, src_lengths, tgt_lengths in pbar:
            # Move to device
            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            output, _ = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
            
            # Loss
            output = output[:, 1:, :].contiguous()
            tgt = tgt[:, 1:].contiguous()
            
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            tgt = tgt.view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)


def train(model, train_loader, val_loader, optimizer, criterion, 
          config, device, save_dir, attention_type):
    
    # Config
    epochs = config['training']['epochs']
    clip = config['training']['gradient_clip']
    patience = config['training']['early_stopping_patience']
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
        min_lr=config['training']['scheduler_min_lr']
    )
    
    # History
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n" + "=" * 70)
    print(f"[INFO] Training Started: {attention_type.upper()} Attention")
    print("=" * 70)
    print(f"   Epochs: {epochs}")
    print(f"   Gradient clip: {clip}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Device: {device}")
    print(f"   Save directory: {save_dir}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, device)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Time
        epoch_time = time.time() - start_time
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)
        
        # Print progress
        print(f"\n[INFO] Epoch {epoch}/{epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Perplexity (optional metric)
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        print(f"   Train PPL: {train_ppl:.2f}")
        print(f"   Val PPL: {val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save checkpoint
            checkpoint_path = save_dir / f"best_model_{attention_type}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"   [OK] Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"   No improvement ({epochs_no_improve}/{patience})")
        
        # Early stopping
        if config['training']['early_stopping'] and epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping triggered (patience={patience})")
            break
    
    # Save final history
    history_path = save_dir / f"history_{attention_type}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"[DONE] Training Completed: {attention_type.upper()}")
    print("=" * 70)
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Best Val PPL: {np.exp(best_val_loss):.2f}")
    print(f"   Total Epochs: {epoch}")
    print(f"   History saved: {history_path}")
    
    return history


def main():

    
    # Argparse
    parser = argparse.ArgumentParser(description='Train Neural Machine Translation Model')
    parser.add_argument('--attention', type=str, required=True,
                       choices=['bahdanau', 'luong', 'scaled_dot'],
                       help='Attention mechanism')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    args = parser.parse_args()
    
    print("=" * 70)
    print("[INFO] Neural Machine Translation - Training")
    print("=" * 70)
    
    # Load config
    print(f"\n[INFO] Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"   [OK] Epochs overridden: {args.epochs}")
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"   [OK] Batch size overridden: {args.batch_size}")
    
    # Set seed
    seed = config['seeds']['torch_seed']
    set_seed(seed)
    
    # Device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\n[INFO] Loading data...")
    data_loader = TranslationDataLoader(args.config)
    data_loader.load_data()
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    src_vocab, tgt_vocab = data_loader.get_vocabs()
    
    # Create model
    print("\n[INFO] Creating model...")
    print(f"   Attention: {args.attention.upper()}")
    
    model = create_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        attention_type=args.attention,
        config=config,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Loss function
    # Padding token'ları ignore et
    pad_idx = tgt_vocab.stoi[tgt_vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print(f"\n  Training configuration:")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Optimizer: Adam")
    print(f"   Loss: CrossEntropyLoss (ignore_index={pad_idx})")
    print(f"   Model parameters: {count_parameters(model):,}")
    
    # Save directory
    save_dir = Path(config['logging']['save_dir']) / args.attention
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config and vocabs
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"\n Config saved: {config_save_path}")
    
    # Save vocabulary info
    vocab_info = {
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'src_pad_idx': src_vocab.stoi[src_vocab.pad_token],
        'src_sos_idx': src_vocab.stoi[src_vocab.sos_token],
        'src_eos_idx': src_vocab.stoi[src_vocab.eos_token],
        'src_unk_idx': src_vocab.stoi[src_vocab.unk_token],
        'tgt_pad_idx': tgt_vocab.stoi[tgt_vocab.pad_token],
        'tgt_sos_idx': tgt_vocab.stoi[tgt_vocab.sos_token],
        'tgt_eos_idx': tgt_vocab.stoi[tgt_vocab.eos_token],
        'tgt_unk_idx': tgt_vocab.stoi[tgt_vocab.unk_token],
    }
    
    vocab_info_path = save_dir / "vocab_info.json"
    with open(vocab_info_path, 'w') as f:
        json.dump(vocab_info, f, indent=2)
    print(f" Vocab info saved: {vocab_info_path}")
    
    # Save full vocabularies
    src_vocab_dict = {'itos': src_vocab.itos, 'stoi': src_vocab.stoi}
    tgt_vocab_dict = {'itos': tgt_vocab.itos, 'stoi': tgt_vocab.stoi}
    
    torch.save(src_vocab_dict, save_dir / "src_vocab.pt")
    torch.save(tgt_vocab_dict, save_dir / "tgt_vocab.pt")
    print(f" Vocabularies saved")
    
    # Train
    history = train(
        model, train_loader, val_loader,
        optimizer, criterion,
        config, device, save_dir, args.attention
    )
    
    print(f"\n Training finished!")
    print(f"   Best model: {save_dir}/best_model_{args.attention}.pt")
    print(f"   History: {save_dir}/history_{args.attention}.json")


if __name__ == "__main__":
    main()

