

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Stil ayarları
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(history_file: Path, output_dir: Path):
    
    # History yükle
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    model_name = history['model_name']
    
    # Figure oluştur
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History: {model_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_accs'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_accs'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=history['best_val_acc'], color='g', linestyle='--', 
               label=f"Best: {history['best_val_acc']:.4f}", alpha=0.7)
    
    # 3. Epoch times
    ax = axes[1, 0]
    ax.bar(epochs, history['epoch_times'], color='skyblue', alpha=0.7)
    ax.axhline(y=history['avg_epoch_time'], color='r', linestyle='--',
               label=f"Avg: {history['avg_epoch_time']:.2f}s")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Epoch Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Convergence
    ax = axes[1, 1]
    ax.plot(epochs, history['val_accs'], 'g-', linewidth=2, marker='o')
    ax.fill_between(epochs, 0, history['val_accs'], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Convergence Efficiency')
    ax.grid(True, alpha=0.3)
    
    # 85% accuracy çizgisi
    ax.axhline(y=0.85, color='r', linestyle='--', label='85% target', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Kaydet
    output_file = output_dir / f"{model_name}_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def plot_model_comparison(experiment_dir: Path, output_dir: Path):
    
    history_files = list(experiment_dir.glob("*_history.json"))
    
    if not history_files:
        print("  Henüz eğitilmiş model yok!")
        return
    
    results = []
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
            results.append(history)
    
  
    df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Val Accuracy': r['best_val_acc'],
            'Train Accuracy': r['train_accs'][-1],
            'Avg Epoch Time (s)': r['avg_epoch_time'],
            'Total Epochs': r['total_epochs']
        }
        for r in results
    ])
    
    # Figure oluştur
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison', fontsize=18, fontweight='bold')
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['Train Accuracy'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, df['Val Accuracy'], width, label='Validation', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Training time comparison
    ax = axes[0, 1]
    bars = ax.bar(df['Model'], df['Avg Epoch Time (s)'], color='coral', alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('Avg Epoch Time (seconds)')
    ax.set_title('Training Speed Comparison')
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar değerlerini yazdır
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # 3. Convergence curves
    ax = axes[1, 0]
    for result in results:
        epochs = range(1, len(result['val_accs']) + 1)
        ax.plot(epochs, result['val_accs'], marker='o', 
                label=result['model_name'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.85, color='r', linestyle='--', alpha=0.5)
    
    # 4. Efficiency scatter
    ax = axes[1, 1]
    scatter = ax.scatter(df['Avg Epoch Time (s)'], df['Val Accuracy'], 
                        s=df['Total Epochs']*20, alpha=0.6, c=range(len(df)),
                        cmap='viridis')
    
    # Model isimlerini ekle
    for i, row in df.iterrows():
        ax.annotate(row['Model'], 
                   (row['Avg Epoch Time (s)'], row['Val Accuracy']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Avg Epoch Time (seconds)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Efficiency vs Performance\n(bubble size = total epochs)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    # Summary table kaydet
    summary_file = output_dir / "model_comparison_table.csv"
    df.to_csv(summary_file, index=False)
    print(f"✓ Saved: {summary_file}")


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path):
    
    plt.figure(figsize=(8, 6))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
   
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Kaydet
    output_file = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def create_summary_report(experiment_dir: Path, output_dir: Path):
   
    history_files = list(experiment_dir.glob("*_history.json"))
    
    if not history_files:
        return
    
    # Markdown rapor
    report_lines = ["# Sentiment Analysis - Model Comparison Report\n"]
    report_lines.append(f"*Generated from {len(history_files)} models*\n")
    report_lines.append("---\n\n")
    
    # Her model için detay
    for history_file in sorted(history_files):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        report_lines.append(f"## {history['model_name']}\n")
        report_lines.append(f"- **Best Validation Accuracy:** {history['best_val_acc']:.4f}\n")
        report_lines.append(f"- **Total Epochs:** {history['total_epochs']}\n")
        report_lines.append(f"- **Avg Epoch Time:** {history['avg_epoch_time']:.2f}s\n")
        report_lines.append(f"- **Total Training Time:** {sum(history['epoch_times'])/60:.2f} min\n")
        
        # Convergence
        for i, acc in enumerate(history['val_accs'], 1):
            if acc >= 0.85:
                report_lines.append(f"- **85% Accuracy at:** Epoch {i}\n")
                break
        
        report_lines.append("\n")
    
    # Kaydet
    report_file = output_dir / "summary_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"✓ Saved: {report_file}")


def main():
    
    parser = argparse.ArgumentParser(description='Visualize training results')
    
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiment klasörü')
    parser.add_argument('--output_dir', type=str, default='experiments/plots',
                       help='Grafiklerin kaydedileceği klasör')
    parser.add_argument('--model', type=str, default=None,
                       help='Belirli bir model için visualization (None = tümü)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION")
    print(f"{'='*60}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    
    if args.model:
        # Tek model
        history_file = experiment_dir / f"{args.model}_history.json"
        if history_file.exists():
            plot_training_curves(history_file, output_dir)
        else:
            print(f"  Model history bulunamadı: {history_file}")
    else:
        # Tüm modeller
        history_files = list(experiment_dir.glob("*_history.json"))
        
        if not history_files:
            print("  Henüz eğitilmiş model yok!")
            return
        
        print(f"Found {len(history_files)} models\n")
        
        # Her model için curves
        for history_file in history_files:
            plot_training_curves(history_file, output_dir)
        
        # Model karşılaştırması
        plot_model_comparison(experiment_dir, output_dir)
        
        # Summary report
        create_summary_report(experiment_dir, output_dir)
    
    print(f"\n{'='*60}")
    print(" Visualization tamamlandı!")
    print(f"Grafikler: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

