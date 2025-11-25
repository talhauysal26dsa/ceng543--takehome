import sys
import os
import argparse
import json
from pathlib import Path

# Redirect output to file IMMEDIATELY to avoid Windows PowerShell encoding issues
LOG_FILE = open("evaluation_output.log", "w", encoding="utf-8", buffering=1)
sys.stdout = LOG_FILE
sys.stderr = LOG_FILE

print("Log file opened, starting imports...")
sys.stdout.flush()

# NOTE: sacrebleu has encoding issues with Windows PowerShell
# Using NLTK's BLEU implementation instead
try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    print("NLTK BLEU imported (sacrebleu causes PowerShell encoding issues)")
    sys.stdout.flush()
    USE_SACREBLEU = False
except:
    try:
        import sacrebleu
        print("Sacrebleu imported")
        sys.stdout.flush()
        USE_SACREBLEU = True
    except Exception as e:
        print(f"ERROR: Could not import BLEU scorer: {e}")
        LOG_FILE.close()
        sys.exit(1)

from rouge_score import rouge_scorer
import torch
import yaml
from tqdm import tqdm

from utils.data_loader import TranslationDataLoader
from models.seq2seq import create_seq2seq_model
from models.transformer import TransformerNMT

print("All imports complete!")
sys.stdout.flush()

def decode_tokens(ids, vocab):
        if hasattr(vocab, "decode"):
        return vocab.decode(ids, skip_special_tokens=True)
    special = {vocab.stoi[vocab.pad_token], vocab.stoi[vocab.sos_token], vocab.stoi[vocab.eos_token]}
    return " ".join([vocab.itos[i] for i in ids if i not in special and i < len(vocab.itos)])

def get_special_ids(vocab):
        if hasattr(vocab, "tokenizer"):
        tokenizer = vocab.tokenizer
        sos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        pad_id = vocab.pad_id
        if sos_id is None or eos_id is None:
            raise ValueError("Tokenizer is missing CLS/SEP (or BOS/EOS) token ids.")
        return sos_id, eos_id, pad_id
    return (
        vocab.stoi[vocab.sos_token],
        vocab.stoi[vocab.eos_token],
        vocab.stoi[vocab.pad_token],
    )

def main():
    print("="*70)
    print("MACHINE TRANSLATION MODEL EVALUATION")
    print("="*70)
    
    parser = argparse.ArgumentParser(description="Evaluate MT models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["seq2seq", "transformer"], required=True, help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Model Type: {args.model}")
    print(f"  Config File: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)

    # Load config
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print("\n[OK] Configuration loaded")
    except Exception as e:
        print(f"\n[ERROR] Failed to load config: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("\n[INFO] Loading test data...")
    try:
        data = TranslationDataLoader(args.config).load_data()
        test_loader = data.get_test_loader()
        src_vocab, tgt_vocab = data.get_vocabs()
        print(f"[OK] Data loaded - {len(test_loader.dataset)} test samples")
        print(f"     Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create model
    print(f"\n[INFO] Creating {args.model} model...")
    try:
        pad_idx = tgt_vocab.pad_id if hasattr(tgt_vocab, "pad_id") else tgt_vocab.stoi[tgt_vocab.pad_token]
        src_pad_idx = src_vocab.pad_id if hasattr(src_vocab, "pad_id") else src_vocab.stoi[src_vocab.pad_token]
        if args.model == "seq2seq":
            model = create_seq2seq_model(len(src_vocab), len(tgt_vocab), config, device, src_pad_idx=src_pad_idx, tgt_pad_idx=pad_idx)
        else:
            model = TransformerNMT(len(src_vocab), len(tgt_vocab), config, pad_idx=pad_idx).to(device)
        print("[OK] Model created")
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load checkpoint
    print(f"\n[INFO] Loading checkpoint...")
    try:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            return
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print("[OK] Checkpoint loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize evaluation
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70 + "\n")
    
    try:
        rouge = rouge_scorer.RougeScorer(config["evaluation"]["rouge_types"], use_stemmer=True)
        rouge_totals = {k: 0.0 for k in config["evaluation"]["rouge_types"]}
    except Exception as e:
        print(f"[ERROR] Failed to initialize ROUGE: {e}")
        return
    
    refs = []
    hyps = []
    count = 0

    # Evaluate
    with torch.no_grad():
        sos_idx, eos_idx, _ = get_special_ids(tgt_vocab)
        for src, tgt, src_lengths, _ in tqdm(test_loader, desc="Evaluating", ncols=80):
            src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)

            # Both models now use translate for proper autoregressive decoding
            batch_size = src.size(0)
            for i in range(batch_size):
                src_i = src[i : i + 1]
                src_len_i = src_lengths[i : i + 1]
                tokens, _ = model.translate(
                    src_i,
                    src_len_i,
                    max_length=config["evaluation"]["max_decode_length"],
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                )
                pred_text = decode_tokens(tokens, tgt_vocab)
                reference = decode_tokens(tgt[i].tolist(), tgt_vocab)
                refs.append(reference)
                hyps.append(pred_text)
                count += 1
                scores = rouge.score(reference, pred_text)
                for k in rouge_totals:
                    rouge_totals[k] += scores[k].fmeasure

    # Compute final metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    try:
        # Compute BLEU
        if USE_SACREBLEU:
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        else:
            # Convert to NLTK format: refs should be list of references for each hypothesis
            # Each reference should be tokenized
            refs_nltk = [[ref.split()] for ref in refs]
            hyps_nltk = [hyp.split() for hyp in hyps]
            smoothing = SmoothingFunction().method1
            bleu = corpus_bleu(refs_nltk, hyps_nltk, smoothing_function=smoothing) * 100
        
        rouge_avg = {k: v / count for k, v in rouge_totals.items()}
        
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"\nModel: {args.model}")
        print(f"Samples: {count}")
        print(f"\nBLEU Score: {bleu:.2f}")
        print(f"\nROUGE Scores:")
        for k, v in rouge_avg.items():
            print(f"  {k.upper()}: {v:.4f}")
        print(f"{'='*70}\n")
        
        # Save results
        out = {"model": args.model, "bleu": bleu, "rouge": rouge_avg, "samples": count}
        save_path = Path(args.checkpoint).with_name("metrics.json")
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Metrics saved to: {save_path}")
        print("\n[SUCCESS] Evaluation completed!\n")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to compute metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        LOG_FILE.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        LOG_FILE.close()
