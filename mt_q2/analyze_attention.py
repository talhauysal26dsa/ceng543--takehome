
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from models.seq2seq import create_seq2seq_model
from utils.data_loader import TranslationDataLoader


def load_model_and_vocab(attention_type, device):
    
    exp_dir = Path(f"experiments/{attention_type}")
    checkpoint_path = exp_dir / f"best_model_{attention_type}.pt"
    config_path = exp_dir / "config.yaml"
    src_vocab_path = exp_dir / "src_vocab.pt"
    tgt_vocab_path = exp_dir / "tgt_vocab.pt"
    
    # Load
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    
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
    
    model = create_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        attention_type=attention_type,
        config=config,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, src_vocab, tgt_vocab, config


def calculate_entropy(attention_weights):
   
    attention_weights = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-10)
    
    # Entropy hesapla (her satır için)
    entropy_values = -np.sum(
        attention_weights * np.log(attention_weights + 1e-10),
        axis=1
    )
    
    mean_entropy = np.mean(entropy_values)
    
    return entropy_values, mean_entropy


def calculate_sharpness(attention_weights):
    
    # Her satırın maksimum değeri
    sharpness_values = np.max(attention_weights, axis=1)
    mean_sharpness = np.mean(sharpness_values)
    
    return sharpness_values, mean_sharpness


def calculate_alignment_entropy(attention_weights):
   
    # Column-wise (her source kelimesi için)
  
    attention_weights_col = attention_weights / (attention_weights.sum(axis=0, keepdims=True) + 1e-10)
    
    # Entropy
    source_entropy = -np.sum(
        attention_weights_col * np.log(attention_weights_col + 1e-10),
        axis=0
    )
    
    return source_entropy


def analyze_single_model(attention_type, device, num_samples=None, test_data_path='data/processed'):
    
    print(f"\n{'='*70}")
    print(f"[INFO] Analyzing: {attention_type.upper()} Attention")
    print(f"{'='*70}")
    
    # Load model
    model, src_vocab, tgt_vocab, config = load_model_and_vocab(attention_type, device)
    
    # Load test data
    print("\n[INFO] Loading test data...")
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from utils.data_loader import TranslationDataset
    from utils.preprocessing import collate_fn
    
    # Load Multi30k test split
    hf_dataset = load_dataset("bentrevett/multi30k")
    test_data = [(item["en"], item["de"]) for item in hf_dataset["test"]][:num_samples] if num_samples else [(item["en"], item["de"]) for item in hf_dataset["test"]]
    
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper
    )
    
    # Metrics
    all_entropies = []
    all_sharpness = []
    all_source_entropies = []
    
    print("\n[INFO] Calculating attention metrics...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Analyzing")):
            if num_samples and i >= num_samples:
                break
            
            # Batch format: (src, tgt, src_lengths, tgt_lengths)
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            
            # Translate and get attention
            sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
            eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
            
            pred_tokens, attention_weights = model.translate(
                src,
                src_lengths,
                tgt_vocab=tgt_vocab,
                max_length=50,
                sos_idx=sos_idx,
                eos_idx=eos_idx
            )
            
            # Skip if too short
            if len(pred_tokens) < 2 or attention_weights.size(0) < 1:
                continue
            
            # Convert to numpy
            attention_np = attention_weights.cpu().numpy()
            
            # Calculate metrics
            entropy_vals, mean_entropy = calculate_entropy(attention_np)
            sharpness_vals, mean_sharpness = calculate_sharpness(attention_np)
            source_entropy = calculate_alignment_entropy(attention_np)
            
            # Save
            all_entropies.append(mean_entropy)
            all_sharpness.append(mean_sharpness)
            all_source_entropies.extend(source_entropy.tolist())
    
    # Statistics
    results = {
        'attention_type': attention_type,
        'num_samples': len(all_entropies),
        'entropy': {
            'mean': float(np.mean(all_entropies)),
            'std': float(np.std(all_entropies)),
            'min': float(np.min(all_entropies)),
            'max': float(np.max(all_entropies)),
            'median': float(np.median(all_entropies)),
            'values': [float(x) for x in all_entropies]
        },
        'sharpness': {
            'mean': float(np.mean(all_sharpness)),
            'std': float(np.std(all_sharpness)),
            'min': float(np.min(all_sharpness)),
            'max': float(np.max(all_sharpness)),
            'median': float(np.median(all_sharpness)),
            'values': [float(x) for x in all_sharpness]
        },
        'source_entropy': {
            'mean': float(np.mean(all_source_entropies)),
            'std': float(np.std(all_source_entropies)),
            'median': float(np.median(all_source_entropies))
        }
    }
    
    # Print summary
    print("\n[INFO] Results:")
    print(f"   Samples analyzed: {results['num_samples']}")
    print(f"   Mean Entropy:     {results['entropy']['mean']:.4f} +/- {results['entropy']['std']:.4f}")
    print(f"   Mean Sharpness:   {results['sharpness']['mean']:.4f} +/- {results['sharpness']['std']:.4f}")
    print(f"   Source Entropy:   {results['source_entropy']['mean']:.4f}")
    
    # Interpretation
    print("\n[INFO] Interpretation:")
    if results['entropy']['mean'] < 1.0:
        print("   [OK] VERY SHARP attention (strongly focused on specific words)")
    elif results['entropy']['mean'] < 1.5:
        print("   [OK] SHARP attention (focused on few words)")
    elif results['entropy']['mean'] < 2.0:
        print("   [INFO] MODERATE attention (distributed across several words)")
    else:
        print("   [WARN] DIFFUSE attention (spread across many words)")
    
    if results['sharpness']['mean'] > 0.7:
        print("   [OK] HIGH sharpness (peak attention weight > 0.7)")
    elif results['sharpness']['mean'] > 0.5:
        print("   [INFO] MODERATE sharpness")
    else:
        print("   [WARN] LOW sharpness (attention spread out)")
    
    # Save results
    exp_dir = Path(f"experiments/{attention_type}")
    results_path = exp_dir / "attention_analysis.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved: {results_path}")
    
    return results


def plot_entropy_distribution(results_list, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Entropy distribution
    ax1 = axes[0]
    for results in results_list:
        att_type = results['attention_type']
        entropy_values = results['entropy']['values']
        ax1.hist(entropy_values, bins=30, alpha=0.6, label=att_type.upper())
    
    ax1.set_xlabel('Entropy', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Attention Entropy Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sharpness distribution
    ax2 = axes[1]
    for results in results_list:
        att_type = results['attention_type']
        sharpness_values = results['sharpness']['values']
        ax2.hist(sharpness_values, bins=30, alpha=0.6, label=att_type.upper())
    
    ax2.set_xlabel('Sharpness (Max Weight)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Attention Sharpness Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Distribution plot saved: {save_path}")
    
    plt.close()


def plot_comparison_metrics(results_list, save_path=None):
    
    attention_types = [r['attention_type'].upper() for r in results_list]
    
    entropy_means = [r['entropy']['mean'] for r in results_list]
    entropy_stds = [r['entropy']['std'] for r in results_list]
    
    sharpness_means = [r['sharpness']['mean'] for r in results_list]
    sharpness_stds = [r['sharpness']['std'] for r in results_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Entropy comparison
    ax1 = axes[0]
    x_pos = np.arange(len(attention_types))
    ax1.bar(x_pos, entropy_means, yerr=entropy_stds, capsize=5, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(attention_types)
    ax1.set_ylabel('Mean Entropy', fontsize=12)
    ax1.set_title('Average Attention Entropy', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(entropy_means):
        ax1.text(i, v + entropy_stds[i] + 0.05, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Sharpness comparison
    ax2 = axes[1]
    ax2.bar(x_pos, sharpness_means, yerr=sharpness_stds, capsize=5,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(attention_types)
    ax2.set_ylabel('Mean Sharpness', fontsize=12)
    ax2.set_title('Average Attention Sharpness', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sharpness_means):
        ax2.text(i, v + sharpness_stds[i] + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Comparison plot saved: {save_path}")
    
    plt.close()


def correlate_with_performance(results_list):
   
    print(f"\n{'='*70}")
    print(f"[INFO] Performance Correlation Analysis")
    print(f"{'='*70}")
    
    # Load evaluation results
    attention_types = []
    entropies = []
    sharpness = []
    bleu_scores = []
    perplexities = []
    
    for results in results_list:
        att_type = results['attention_type']
        
        # Try to load evaluation results
        eval_path = Path(f"experiments/{att_type}/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            
            attention_types.append(att_type.upper())
            entropies.append(results['entropy']['mean'])
            sharpness.append(results['sharpness']['mean'])
            bleu_scores.append(eval_results.get('bleu', None))
            perplexities.append(eval_results.get('perplexity', None))
    
    # Check if we have data
    if not bleu_scores or all(b is None for b in bleu_scores):
        print("   WARNING: No evaluation results found. Run evaluate.py first.")
        return None
    
    # Correlation analysis
    print("\n[INFO] Correlation Analysis:")
    print(f"{'Metric Pair':<40} {'Pearson r':<12} {'Spearman rho':<12}")
    print("-" * 70)
    
    correlations = {}
    
    # Entropy vs BLEU
    if None not in bleu_scores:
        pearson_r, p_val = pearsonr(entropies, bleu_scores)
        spearman_r, sp_val = spearmanr(entropies, bleu_scores)
        print(f"{'Entropy vs BLEU':<40} {pearson_r:>+.4f}       {spearman_r:>+.4f}")
        correlations['entropy_bleu'] = {
            'pearson': pearson_r,
            'spearman': spearman_r
        }
    
    # Sharpness vs BLEU
    if None not in bleu_scores:
        pearson_r, p_val = pearsonr(sharpness, bleu_scores)
        spearman_r, sp_val = spearmanr(sharpness, bleu_scores)
        print(f"{'Sharpness vs BLEU':<40} {pearson_r:>+.4f}       {spearman_r:>+.4f}")
        correlations['sharpness_bleu'] = {
            'pearson': pearson_r,
            'spearman': spearman_r
        }
    
    # Entropy vs Perplexity
    if None not in perplexities:
        pearson_r, p_val = pearsonr(entropies, perplexities)
        spearman_r, sp_val = spearmanr(entropies, perplexities)
        print(f"{'Entropy vs Perplexity':<40} {pearson_r:>+.4f}       {spearman_r:>+.4f}")
        correlations['entropy_perplexity'] = {
            'pearson': pearson_r,
            'spearman': spearman_r
        }
    
    # Interpretation
    print("\n[INFO] Interpretation:")
    print(f"   r > 0.7:  Strong positive correlation")
    print(f"   r > 0.4:  Moderate positive correlation")
    print(f"   r < -0.4: Moderate negative correlation")
    print(f"   r < -0.7: Strong negative correlation")
    
    return correlations


def compare_all_models(results_list):
   
    print(f"\n{'='*70}")
    print(f"[SUMMARY] ATTENTION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    # Table header
    print(f"\n{'Attention':<15} {'Entropy':<15} {'Sharpness':<15} {'Source Entropy':<15}")
    print("-" * 70)
    
    # Results
    for results in results_list:
        att = results['attention_type'].upper()
        ent = f"{results['entropy']['mean']:.4f} ± {results['entropy']['std']:.4f}"
        sharp = f"{results['sharpness']['mean']:.4f} ± {results['sharpness']['std']:.4f}"
        src_ent = f"{results['source_entropy']['mean']:.4f}"
        
        print(f"{att:<15} {ent:<15} {sharp:<15} {src_ent:<15}")
    
    # Find extremes
    print(f"\n{'='*70}")
    print(f"[SUMMARY] RANKINGS")
    print(f"{'='*70}")
    
    # Lowest entropy (most focused)
    min_entropy = min(results_list, key=lambda x: x['entropy']['mean'])
    print(f"   Most Focused (Lowest Entropy):   {min_entropy['attention_type'].upper()} "
          f"({min_entropy['entropy']['mean']:.4f})")
    
    # Highest sharpness
    max_sharpness = max(results_list, key=lambda x: x['sharpness']['mean'])
    print(f"   Sharpest (Highest Sharpness):    {max_sharpness['attention_type'].upper()} "
          f"({max_sharpness['sharpness']['mean']:.4f})")
    
    # Save comparison
    comparison_path = Path("experiments/attention_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump({
            'results': results_list,
            'most_focused': min_entropy['attention_type'],
            'sharpest': max_sharpness['attention_type']
        }, f, indent=2)
    
    print(f"\n[INFO] Comparison saved: {comparison_path}")


def main():
    """Ana analysis fonksiyonu."""
    
    # Argparse
    parser = argparse.ArgumentParser(description='Analyze Attention Mechanisms')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to analyze (None = all)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n[INFO] Device: {device}")
    
    # Analyze all models
    attention_types = ['bahdanau', 'luong', 'scaled_dot']
    results_list = []
    
    for attention_type in attention_types:
        results = analyze_single_model(
            attention_type,
            device,
            num_samples=args.num_samples
        )
        results_list.append(results)
    
    # Compare all
    compare_all_models(results_list)
    
    # Correlate with performance
    correlate_with_performance(results_list)
    
    # Visualizations
    print("\n[INFO] Creating visualizations...")
    
    output_dir = Path("experiments/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_entropy_distribution(
        results_list,
        save_path=output_dir / "entropy_distribution.png"
    )
    
    plot_comparison_metrics(
        results_list,
        save_path=output_dir / "metrics_comparison.png"
    )
    
    print(f"\n{'='*70}")
    print(f"[DONE] ATTENTION ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\n[INFO] Results saved in 'experiments/<attention_type>/attention_analysis.json'")
    print("[INFO] Visualizations saved in 'experiments/visualizations/'")


if __name__ == "__main__":
    main()

