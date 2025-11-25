

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict

# BLEU ve ROUGE için gerekli kütüphaneler
try:
    from sacrebleu import corpus_bleu, sentence_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    print("WARNING: sacrebleu not available. Install with: pip install sacrebleu")
    SACREBLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("WARNING: rouge_score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

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
    
    # Check files
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("   [OK] Config loaded")
    
    # Load vocabularies (saved as dicts with 'itos' and 'stoi')
    src_vocab_data = torch.load(src_vocab_path)
    tgt_vocab_data = torch.load(tgt_vocab_path)

    def rebuild_vocab(vocab_data):
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
    
    print(f"   [OK] Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
    
    return model, src_vocab, tgt_vocab, config


def translate_sentence(model, sentence_tokens, src_vocab, tgt_vocab, device, max_length=50):
   
    model.eval()
    
    with torch.no_grad():
        # Tokenları ID'lere çevir
        src_indices = [src_vocab[token] for token in sentence_tokens]
        
        # Tensor'e çevir
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
        src_lengths = torch.tensor([len(src_indices)]).to(device)
        
        # Translate
        sos_idx = tgt_vocab.stoi[tgt_vocab.sos_token]
        eos_idx = tgt_vocab.stoi[tgt_vocab.eos_token]
        
        pred_tokens, attention_weights = model.translate(
            src_tensor,
            src_lengths,
            tgt_vocab=tgt_vocab,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )
        
        # ID'leri kelimelere çevir (<sos> ve <eos> hariç)
        translation = []
        for token_id in pred_tokens[1:]:  # İlk <sos> atla
            if token_id == eos_idx:
                break
            translation.append(tgt_vocab.itos[token_id])
        
        return translation, attention_weights


def calculate_perplexity(model, dataloader, criterion, device):
   
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity", leave=False):
            # Batch format: (src, tgt, src_lengths, tgt_lengths)
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            tgt = tgt.to(device)
            
            # Forward
            outputs, _ = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            
            # Loss (reshape)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)  # [batch*tgt_len, vocab_size]
            tgt = tgt[:, 1:].reshape(-1)  # [batch*tgt_len]
            
            # Calculate loss
            loss = criterion(outputs, tgt)
            
            # Count non-padding tokens
            non_pad_tokens = (tgt != 0).sum().item()
            
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_bleu_rouge(model, dataloader, src_vocab, tgt_vocab, device, num_samples=None):
    
    model.eval()
    
    references = []  # [[ref1], [ref2], ...] (her biri list of strings)
    hypotheses = []  # [hyp1, hyp2, ...] (her biri string)
    
    # ROUGE scorer
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    print("\n[INFO] Generating translations...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Translating")):
            if num_samples and i >= num_samples:
                break
            
            # Batch format: (src, tgt, src_lengths, tgt_lengths)
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            tgt = tgt.to(device)
            
            batch_size = src.size(0)
            
            # Her örnek için translate et
            for j in range(batch_size):
                # Source tokens
                src_tokens = []
                for idx in src[j]:
                    idx = idx.item()
                    if idx == 0:  # <pad>
                        break
                    src_tokens.append(src_vocab.itos[idx])
                
                # Reference translation
                ref_tokens = []
                for idx in tgt[j][1:]:  # <sos> atla
                    idx = idx.item()
                    if idx == tgt_vocab.stoi[tgt_vocab.eos_token]:
                        break
                    if idx == 0:  # <pad>
                        continue
                    ref_tokens.append(tgt_vocab.itos[idx])
                
                # Model translation
                pred_tokens, _ = translate_sentence(
                    model, src_tokens, src_vocab, tgt_vocab, device
                )
                
                # String'e çevir
                ref_str = " ".join(ref_tokens)
                pred_str = " ".join(pred_tokens)
                
                references.append([ref_str])  # BLEU için list of list
                hypotheses.append(pred_str)
    
    print(f"   [OK] {len(hypotheses)} translations generated")
    
    # Calculate BLEU
    results = {}
    
    if SACREBLEU_AVAILABLE:
        # Corpus BLEU - sacrebleu expects references as list of lists
        # Each reference list should contain one string per reference translation
        refs_formatted = [[ref[0]] for ref in references]  # [[ref1], [ref2], ...]
        refs_transposed = list(map(list, zip(*refs_formatted)))  # Transpose
        
        bleu_score = corpus_bleu(hypotheses, refs_transposed).score
        results['bleu'] = bleu_score
        print(f"   [OK] BLEU: {bleu_score:.2f}")
    else:
        print("   WARNING: BLEU not calculated (sacrebleu not available)")
        results['bleu'] = None
    
    # Calculate ROUGE
    if ROUGE_AVAILABLE:
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref_list, hyp in zip(references, hypotheses):
            ref = ref_list[0]  # İlk reference'ı kullan
            scores = scorer.score(ref, hyp)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Average
        results['rouge1'] = np.mean(rouge_scores['rouge1'])
        results['rouge2'] = np.mean(rouge_scores['rouge2'])
        results['rougeL'] = np.mean(rouge_scores['rougeL'])
        
        print(f"   [OK] ROUGE-1: {results['rouge1']:.4f}")
        print(f"   [OK] ROUGE-2: {results['rouge2']:.4f}")
        print(f"   [OK] ROUGE-L: {results['rougeL']:.4f}")
    else:
        print("   WARNING: ROUGE not calculated (rouge_score not available)")
        results['rouge1'] = None
        results['rouge2'] = None
        results['rougeL'] = None
    
    # Sample translations
    results['sample_translations'] = []
    for i in range(min(5, len(references))):
        results['sample_translations'].append({
            'reference': references[i][0],
            'hypothesis': hypotheses[i]
        })
    
    return results


def evaluate_model(attention_type, device, test_data_path='data/processed', num_samples=None):
    
    print("\n" + "=" * 70)
    print(f"[INFO] Evaluating: {attention_type.upper()} Attention")
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
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper
    )
    
    print(f"   [OK] Test samples: {len(test_dataset)}")
    
    # 1. Perplexity
    print("\n[INFO] Calculating Perplexity...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    perplexity = calculate_perplexity(model, test_loader, criterion, device)
    print(f"   [OK] Perplexity: {perplexity:.2f}")
    
    # 2. BLEU & ROUGE
    bleu_rouge_results = calculate_bleu_rouge(
        model, test_loader, src_vocab, tgt_vocab, device, num_samples
    )
    
    # Combine results
    results = {
        'attention_type': attention_type,
        'perplexity': perplexity,
        'bleu': bleu_rouge_results.get('bleu'),
        'rouge1': bleu_rouge_results.get('rouge1'),
        'rouge2': bleu_rouge_results.get('rouge2'),
        'rougeL': bleu_rouge_results.get('rougeL'),
        'sample_translations': bleu_rouge_results.get('sample_translations', [])
    }
    
    # Save results
    exp_dir = Path(f"experiments/{attention_type}")
    results_path = exp_dir / "evaluation_results.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Results saved: {results_path}")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print(f"[DONE] Evaluation Complete: {attention_type.upper()}")
    print("=" * 70)
    print(f"   Perplexity:  {perplexity:.2f}")
    if results['bleu'] is not None:
        print(f"   BLEU:        {results['bleu']:.2f}")
    if results['rougeL'] is not None:
        print(f"   ROUGE-L:     {results['rougeL']:.4f}")
    
    return results


def compare_models(results_list):
    
    print("\n" + "=" * 70)
    print("[SUMMARY] MODEL COMPARISON")
    print("=" * 70)
    
    # Header
    print(f"\n{'Attention Type':<20} {'Perplexity':<12} {'BLEU':<10} {'ROUGE-L':<10}")
    print("-" * 70)
    
    # Results
    for results in results_list:
        att_type = results['attention_type'].upper()
        ppl = results['perplexity']
        bleu = results['bleu'] if results['bleu'] is not None else -1
        rouge = results['rougeL'] if results['rougeL'] is not None else -1
        
        bleu_str = f"{bleu:.2f}" if bleu >= 0 else "N/A"
        rouge_str = f"{rouge:.4f}" if rouge >= 0 else "N/A"
        
        print(f"{att_type:<20} {ppl:<12.2f} {bleu_str:<10} {rouge_str:<10}")
    
    # Find best
    print("\n" + "=" * 70)
    print("[SUMMARY] BEST MODELS")
    print("=" * 70)
    
    # Best perplexity (lowest)
    best_ppl = min(results_list, key=lambda x: x['perplexity'])
    print(f"   Lowest Perplexity:  {best_ppl['attention_type'].upper()} ({best_ppl['perplexity']:.2f})")
    
    # Best BLEU (highest)
    valid_bleu = [r for r in results_list if r['bleu'] is not None]
    if valid_bleu:
        best_bleu = max(valid_bleu, key=lambda x: x['bleu'])
        print(f"   Highest BLEU:       {best_bleu['attention_type'].upper()} ({best_bleu['bleu']:.2f})")
    
    # Best ROUGE-L (highest)
    valid_rouge = [r for r in results_list if r['rougeL'] is not None]
    if valid_rouge:
        best_rouge = max(valid_rouge, key=lambda x: x['rougeL'])
        print(f"   Highest ROUGE-L:    {best_rouge['attention_type'].upper()} ({best_rouge['rougeL']:.4f})")
    
    # Save comparison
    comparison_path = Path("experiments/model_comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results_list,
            'best_perplexity': best_ppl['attention_type'],
            'best_bleu': best_bleu['attention_type'] if valid_bleu else None,
            'best_rouge': best_rouge['attention_type'] if valid_rouge else None
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Comparison saved: {comparison_path}")


def main():
    
    
    # Argparse
    parser = argparse.ArgumentParser(description='Evaluate Neural Machine Translation Models')
    parser.add_argument('--attention', type=str, default='all',
                       choices=['bahdanau', 'luong', 'scaled_dot', 'all'],
                       help='Which model to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n[INFO] Device: {device}")
    
    # Evaluate
    if args.attention == 'all':
        attention_types = ['bahdanau', 'luong', 'scaled_dot']
    else:
        attention_types = [args.attention]
    
    results_list = []
    for attention_type in attention_types:
        results = evaluate_model(attention_type, device, num_samples=args.num_samples)
        results_list.append(results)
    
    # Compare if multiple models
    if len(results_list) > 1:
        compare_models(results_list)
    
    print("\n" + "=" * 70)
    print("[DONE] EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

