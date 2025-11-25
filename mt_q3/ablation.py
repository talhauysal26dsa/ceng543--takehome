import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

def run_ablation(base_config_path: str, layers_list, heads_list, model_type: str, epochs: int = None):
        from train import main as train_main
    
    results = []
    
    for layers in layers_list:
        for heads in heads_list:
            tag = f"layers{layers}_heads{heads}"
            print(f"\n{'='*70}")
            print(f"[Ablation] Starting: {tag}")
            print(f"{'='*70}")
            
            # Create temporary config with modified parameters
            tmp_config = f".tmp_config_{tag}.yaml"
            with open(base_config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Modify transformer config
            config["transformer"]["num_encoder_layers"] = layers
            config["transformer"]["num_decoder_layers"] = layers
            config["transformer"]["nhead"] = heads
            
            # Override save directory for ablation
            if "logging" not in config:
                config["logging"] = {}
            config["logging"]["save_dir"] = f"experiments/ablation/{tag}"
            
            # Save temporary config
            with open(tmp_config, "w") as f:
                yaml.dump(config, f)
            
            # Prepare arguments for train.py
            sys.argv = [
                "ablation.py",
                "--config", tmp_config,
                "--model", model_type,
            ]
            if epochs:
                sys.argv.extend(["--epochs", str(epochs)])
            
            try:
                # Run training
                train_main()
                
                # Load results
                result_dir = Path(config["logging"]["save_dir"]) / model_type
                history_file = result_dir / "history.json"
                if history_file.exists():
                    with open(history_file, "r") as f:
                        history = json.load(f)
                    best_val_loss = min(history["val_loss"])
                    results.append({
                        "layers": layers,
                        "heads": heads,
                        "best_val_loss": best_val_loss,
                        "config": tag
                    })
                    print(f"\n[Ablation] {tag} completed - Best val loss: {best_val_loss:.4f}")
                
            except Exception as e:
                print(f"\n[ERROR] {tag} failed: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Cleanup temporary config
                Path(tmp_config).unlink(missing_ok=True)
    
    # Save ablation results
    ablation_dir = Path("experiments/ablation")
    ablation_dir.mkdir(parents=True, exist_ok=True)
    results_file = ablation_dir / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETED")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_file}")
    print("\nSummary:")
    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        print(f"  {r['config']}: val_loss={r['best_val_loss']:.4f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for Transformer architectures")
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config file")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer"], 
                        help="Model type (only transformer supported for ablation)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    layers_list = config.get("ablation", {}).get("layers", [2, 4, 6])
    heads_list = config.get("ablation", {}).get("heads", [2, 4, 8])
    
    print(f"\n{'='*70}")
    print("TRANSFORMER ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Base config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Layers to test: {layers_list}")
    print(f"Heads to test: {heads_list}")
    print(f"Total configurations: {len(layers_list) * len(heads_list)}")
    if args.epochs:
        print(f"Epochs per config: {args.epochs}")
    print(f"{'='*70}\n")
    
    run_ablation(args.config, layers_list, heads_list, args.model, args.epochs)
