

import json
from pathlib import Path
import pandas as pd
import torch

from evaluate import load_model_and_data, SentimentEvaluator


def evaluate_all_models(experiment_dir: str = 'experiments', output_file: str = 'full_results.json'):
   
    experiment_path = Path(experiment_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("FULL MODEL EVALUATION")
    print("="*70)
    print(f"Device: {device}\n")
    
    best_checkpoints = list(experiment_path.glob("*/*_best.pth"))
    
    if not best_checkpoints:
        print(" Hiç eğitilmiş model bulunamadı!")
        return
    
    print(f" {len(best_checkpoints)} model bulundu\n")
    
    all_results = {}
    
    for checkpoint_path in best_checkpoints:

        model_name = checkpoint_path.parent.name
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")
        
        model, test_loader, is_bert = load_model_and_data(
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            data_dir='data/raw/aclImdb',
            batch_size=32,
            device=device
        )
        
        if model is None:
            print(f" {model_name} için evaluation atlandı (GloVe vocab problemi)")
            continue
        
        # Evaluate
        evaluator = SentimentEvaluator(model, test_loader, device, is_bert)
        metrics = evaluator.evaluate()
        
        history_file = checkpoint_path.parent / f"{model_name}_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            metrics['training_history'] = {
                'total_epochs': history['total_epochs'],
                'best_val_acc': history['best_val_acc'],
                'avg_epoch_time': history['avg_epoch_time'],
                'convergence_epoch': get_convergence_epoch(history['val_accs'])
            }
        
        all_results[model_name] = metrics
    
    
    output_path = Path(experiment_dir) / output_file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"✓ Tüm sonuçlar kaydedildi: {output_path}")
    print(f"{'='*70}")
    
    print_summary_table(all_results)
    
    return all_results


def get_convergence_epoch(val_accs, threshold=0.85):
    
    for i, acc in enumerate(val_accs, 1):
        if acc >= threshold:
            return i
    return None


def print_summary_table(results):
  
    print("\n\n" + "="*90)
    print("SUMMARY TABLE - ALL METRICS")
    print("="*90)
    
    
    data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Test Accuracy': f"{metrics['accuracy']:.4f}",
            'Macro F1': f"{metrics['macro_f1']:.4f}",
            'F1 Negative': f"{metrics['f1_negative']:.4f}",
            'F1 Positive': f"{metrics['f1_positive']:.4f}",
        }
        
        if 'training_history' in metrics:
            hist = metrics['training_history']
            row['Epochs'] = hist['total_epochs']
            row['Val Acc'] = f"{hist['best_val_acc']:.4f}"
            row['Convergence'] = f"{hist['convergence_epoch']} ep" if hist['convergence_epoch'] else "N/A"
            row['Avg Time/Epoch'] = f"{hist['avg_epoch_time']:.1f}s"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
   
    df = df.sort_values('Test Accuracy', ascending=False)
    
    print(df.to_string(index=False))
    print("="*90)
    
   
    print("\n BEST MODELS:")
    print(f"  Best Accuracy: {df.iloc[0]['Model']} ({df.iloc[0]['Test Accuracy']})")
    print(f"  Best Macro F1: {df.iloc[0]['Model']} ({df.iloc[0]['Macro F1']})")
    
    if 'Convergence' in df.columns:
        fastest = df.loc[df['Convergence'] != 'N/A'].sort_values('Convergence').iloc[0]
        print(f"  Fastest Convergence: {fastest['Model']} ({fastest['Convergence']})")
    
    print("\n")


def create_comparison_report(results, output_file='experiments/comparison_report.md'):
    
    lines = ["# Sentiment Analysis - Full Model Comparison\n\n"]
    lines.append(f"*Evaluated {len(results)} models*\n\n")
    lines.append("---\n\n")
    
    # Her model için detay
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        lines.append(f"## {model_name.upper()}\n\n")
        lines.append(f"### Test Set Performance\n")
        lines.append(f"- **Accuracy:** {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        lines.append(f"- **Macro F1:** {metrics['macro_f1']:.4f}\n")
        lines.append(f"- **F1 Negative:** {metrics['f1_negative']:.4f}\n")
        lines.append(f"- **F1 Positive:** {metrics['f1_positive']:.4f}\n")
        lines.append(f"- **Precision Negative:** {metrics['precision_negative']:.4f}\n")
        lines.append(f"- **Precision Positive:** {metrics['precision_positive']:.4f}\n")
        lines.append(f"- **Recall Negative:** {metrics['recall_negative']:.4f}\n")
        lines.append(f"- **Recall Positive:** {metrics['recall_positive']:.4f}\n\n")
        
        if 'training_history' in metrics:
            hist = metrics['training_history']
            lines.append(f"### Training Information\n")
            lines.append(f"- **Total Epochs:** {hist['total_epochs']}\n")
            lines.append(f"- **Best Val Accuracy:** {hist['best_val_acc']:.4f}\n")
            lines.append(f"- **Avg Time/Epoch:** {hist['avg_epoch_time']:.2f}s\n")
            if hist['convergence_epoch']:
                lines.append(f"- **Convergence (85%):** {hist['convergence_epoch']} epochs\n")
            lines.append("\n")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        lines.append(f"### Confusion Matrix\n")
        lines.append("```\n")
        lines.append(f"              Predicted\n")
        lines.append(f"              Neg    Pos\n")
        lines.append(f"Actual  Neg  [{cm[0][0]:5d}  {cm[0][1]:5d}]\n")
        lines.append(f"        Pos  [{cm[1][0]:5d}  {cm[1][1]:5d}]\n")
        lines.append("```\n\n")
        
        lines.append("---\n\n")
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ Comparison report saved: {output_file}")


if __name__ == "__main__":
    results = evaluate_all_models()
    
    if results:
        
        create_comparison_report(results)
        
        print("\n" + "="*70)
        print(" FULL EVALUATION COMPLETED!")
        print("="*70)
        print("\nSonraki adım: Visualization")
        print("  python visualize_embeddings.py")




