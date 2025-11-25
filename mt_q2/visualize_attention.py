

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from models.seq2seq import create_seq2seq_model
from utils.data_loader import TranslationDataLoader
from utils.preprocessing import Vocabulary


def load_model_and_vocab(attention_type, device):
    
    print(f"\n[INFO] Loading model: {attention_type.upper()}")
    
    # Paths
    exp_dir = Path(f"experiments/{attention_type}")
    checkpoint_path = exp_dir / f"best_model_{attention_type}.pt"
    config_path = exp_dir / "config.yaml"
    src_vocab_path = exp_dir / "src_vocab.pt"
    tgt_vocab_path = exp_dir / "tgt_vocab.pt"
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("   [OK] Config loaded")
    
    # Load vocabularies (saved as dicts with 'itos' and 'stoi')
    src_vocab_data = torch.load(src_vocab_path)
    tgt_vocab_data = torch.load(tgt_vocab_path)

    def rebuild_vocab(vocab_data):
        from utils.preprocessing import Vocabulary
        if isinstance(vocab_data, Vocabulary):
            return vocab_data
        vocab = Vocabulary()
        vocab.itos = vocab_data['itos']
        vocab.stoi = vocab_data['stoi']
        return vocab

    src_vocab = rebuild_vocab(src_vocab_data)
    tgt_vocab = rebuild_vocab(tgt_vocab_data)
    print(f"   [OK] Vocabularies loaded (src size: {len(src_vocab)}, tgt size: {len(tgt_vocab)})")
    
    # Create model
    model = create_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        attention_type=attention_type,
        config=config,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("   [OK] Model loaded")
    
    return model, src_vocab, tgt_vocab, config


def translate_with_attention(model, src_tensor, src_lengths, src_vocab, tgt_vocab, device):
    
    model.eval()
    
    with torch.no_grad():
        # Translate
        sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
        eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
        
        pred_token_ids, attention_weights = model.translate(
            src_tensor,
            src_lengths,
            tgt_vocab=tgt_vocab,
            max_length=50,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )
        
        # Source tokens
        src_tokens = []
        for idx in src_tensor[0]:
            idx = idx.item()
            if idx == 0:  # <pad>
                break
            src_tokens.append(src_vocab.itos[idx])
        
        # Target tokens
        tgt_tokens = []
        for token_id in pred_token_ids[1:]:  # <sos> atla
            if token_id == eos_idx:
                break
            tgt_tokens.append(tgt_vocab.itos[token_id])
        
        # Attention weights (sadece generated token'lar için)
        attention_weights = attention_weights[:len(tgt_tokens), :len(src_tokens)]
        
        return src_tokens, tgt_tokens, attention_weights


def plot_attention(src_tokens, tgt_tokens, attention_weights, 
                   attention_type, save_path=None, show=True):
   
    # Figure boyutu (kelime sayısına göre)
    fig_width = max(8, len(src_tokens) * 0.5)
    fig_height = max(6, len(tgt_tokens) * 0.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Heatmap
    sns.heatmap(
        attention_weights.numpy(),
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Labels
    ax.set_xlabel('Source (English)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target (German)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Attention Alignment Map - {attention_type.upper()} Attention',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Saved: {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()


def visualize_samples(attention_type, device, num_samples=5, 
                      test_data_path='data/processed', show=False):
   
    print("\n" + "=" * 70)
    print(f"[INFO] Visualizing Attention: {attention_type.upper()}")
    print("=" * 70)
    
    # Load model and vocab
    model, src_vocab, tgt_vocab, config = load_model_and_vocab(attention_type, device)
    
    # Load test data
    print("\n[INFO] Loading test data...")
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from utils.data_loader import TranslationDataset
    from utils.preprocessing import collate_fn
    
    # Load Multi30k test split
    hf_dataset = load_dataset("bentrevett/multi30k")
    test_data = [(item["en"], item["de"]) for item in hf_dataset["test"]]
    
    print(f"   [OK] Loaded {len(test_data)} test examples")
    
    # Create dataset
    test_dataset = TranslationDataset(
        test_data, src_vocab, tgt_vocab, max_length=50
    )
    
    # Create dataloader
    pad_idx = src_vocab.stoi[src_vocab.pad_token]
    def collate_wrapper(batch):
        return collate_fn(batch, pad_idx)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Tek tek işle
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper
    )
    
    print(f"   [OK] Test samples: {len(test_dataset)}")
    
    # Create output directory
    output_dir = Path(f"experiments/{attention_type}/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n[INFO] Generating attention maps...")
    
    # Visualize samples
    count = 0
    for batch in tqdm(test_loader, desc="Visualizing", total=num_samples):
        if count >= num_samples:
            break
        
        # Batch format: (src, tgt, src_lengths, tgt_lengths)
        src, tgt, src_lengths, tgt_lengths = batch
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        
        # Translate with attention
        src_tokens, tgt_tokens, attention_weights = translate_with_attention(
            model, src, src_lengths, src_vocab, tgt_vocab, device
        )
        
        # Plot
        save_path = output_dir / f"attention_map_{count + 1}.png"
        plot_attention(
            src_tokens,
            tgt_tokens,
            attention_weights,
            attention_type,
            save_path=save_path,
            show=show
        )
        
        count += 1
    
    print(f"\n[DONE] Visualized {count} samples")
    print(f"   Output directory: {output_dir}")
    
    return output_dir


def create_comparison_visualization(sample_idx=0, test_data_path='data/processed', 
                                   show=False):
   
    print("\n" + "=" * 70)
    print("[INFO] Creating Comparison Visualization")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention_types = ['bahdanau', 'luong', 'scaled_dot']
    
    # Shared data loading (ilk modelden vocab al)
    first_model, src_vocab, tgt_vocab, _ = load_model_and_vocab(attention_types[0], device)
    
    # Load test data
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from utils.data_loader import TranslationDataset
    from utils.preprocessing import collate_fn
    
    # Load Multi30k test split
    hf_dataset = load_dataset("bentrevett/multi30k")
    test_data = [(item["en"], item["de"]) for item in hf_dataset["test"]]
    
    # Create dataset
    test_dataset = TranslationDataset(
        test_data, src_vocab, tgt_vocab, max_length=50
    )
    
    # Create dataloader
    pad_idx = src_vocab.stoi[src_vocab.pad_token]
    def collate_wrapper(batch):
        return collate_fn(batch, pad_idx)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper
    )
    
    # Get sample
    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            # Batch format: (src, tgt, src_lengths, tgt_lengths)
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            break
    
    # Get source tokens (ortak)
    src_tokens = []
    for idx in src[0]:
        idx = idx.item()
        if idx == 0:
            break
        src_tokens.append(src_vocab.itos[idx])
    
    print(f"\n[INFO] Source: {' '.join(src_tokens)}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    attention_data = []
    
    for idx, attention_type in enumerate(attention_types):
        print(f"\n[INFO] Processing: {attention_type.upper()}")
        
        # Load model
        model, _, _, _ = load_model_and_vocab(attention_type, device)
        
        # Translate
        src_tokens_att, tgt_tokens, attention_weights = translate_with_attention(
            model, src, src_lengths, src_vocab, tgt_vocab, device
        )
        
        print(f"   Target: {' '.join(tgt_tokens)}")
        
        attention_data.append({
            'src_tokens': src_tokens_att,
            'tgt_tokens': tgt_tokens,
            'attention_weights': attention_weights,
            'attention_type': attention_type
        })
        
        # Plot in subplot
        ax = axes[idx]
        sns.heatmap(
            attention_weights.numpy(),
            xticklabels=src_tokens_att,
            yticklabels=tgt_tokens,
            cmap='YlOrRd',
            cbar_kws={'label': 'Weight'},
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_xlabel('Source (EN)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Target (DE)', fontsize=10, fontweight='bold')
        ax.set_title(f'{attention_type.upper()} Attention', 
                    fontsize=12, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.suptitle(
        f'Attention Mechanism Comparison (Sample {sample_idx + 1})',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("experiments/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"comparison_sample_{sample_idx + 1}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comparison saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def main():
    """Ana visualization fonksiyonu."""
    
    # Argparse
    parser = argparse.ArgumentParser(description='Visualize Attention Mechanisms')
    parser.add_argument('--attention', type=str, default='all',
                       choices=['bahdanau', 'luong', 'scaled_dot', 'all'],
                       help='Which model to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--comparison', action='store_true',
                       help='Create comparison visualization')
    parser.add_argument('--comparison_sample', type=int, default=0,
                       help='Which sample to use for comparison')
    parser.add_argument('--show', action='store_true',
                       help='Show plots (default: only save)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n[INFO] Device: {device}")
    
    # Visualize individual models
    if args.attention == 'all':
        attention_types = ['bahdanau', 'luong', 'scaled_dot']
    else:
        attention_types = [args.attention]
    
    for attention_type in attention_types:
        visualize_samples(
            attention_type,
            device,
            num_samples=args.num_samples,
            show=args.show
        )
    
    # Comparison visualization
    if args.comparison:
        create_comparison_visualization(
            sample_idx=args.comparison_sample,
            show=args.show
        )
    
    print("\n" + "=" * 70)
    print("[DONE] VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n[INFO] Check 'experiments/<attention_type>/visualizations/' for individual maps")
    print(f"[INFO] Check 'experiments/visualizations/' for comparison maps")


if __name__ == "__main__":
    main()

