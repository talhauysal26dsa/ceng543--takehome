

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_results():
    """Tüm sonuçları yükle."""
    
    results = {}
    attention_types = ['bahdanau', 'luong', 'scaled_dot']
    
    for att_type in attention_types:
        exp_dir = Path(f"experiments/{att_type}")
        
        # Training history
        history_path = exp_dir / f"history_{att_type}.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                results[f'{att_type}_history'] = json.load(f)
        
        # Evaluation results
        eval_path = exp_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                results[f'{att_type}_eval'] = json.load(f)
        
        # Attention analysis
        analysis_path = exp_dir / "attention_analysis.json"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                results[f'{att_type}_analysis'] = json.load(f)
    
    return results


def generate_markdown_report(results, output_path):
    """
    Markdown format rapor oluştur.
    
    Args:
        results: Dict with all results
        output_path: Output file path
    """
    
    attention_types = ['bahdanau', 'luong', 'scaled_dot']
    
    # Start markdown
    md = []
    md.append("# Machine Translation - Attention Mechanisms Comparison Report\n")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("---\n\n")
    
    # 1. Executive Summary
    md.append("##  Executive Summary\n\n")
    md.append("This report presents a comprehensive comparison of three attention mechanisms ")
    md.append("for neural machine translation (English→German) using the Multi30k dataset:\n\n")
    md.append("1. **Bahdanau (Additive) Attention** - Original attention mechanism (2015)\n")
    md.append("2. **Luong (Multiplicative) Attention** - Simplified version (2015)\n")
    md.append("3. **Scaled Dot-Product Attention** - Used in Transformers (2017)\n\n")
    md.append("All models were trained with identical hyperparameters and random seeds ")
    md.append("to ensure fair comparison.\n\n")
    md.append("---\n\n")
    
    # 2. Performance Metrics
    md.append("##  Performance Metrics\n\n")
    
    # Check if evaluation results exist
    has_eval = any(f'{att}_eval' in results for att in attention_types)
    
    if has_eval:
        md.append("### Translation Quality\n\n")
        md.append("| Attention Type | BLEU ↑ | ROUGE-1 ↑ | ROUGE-L ↑ | Perplexity ↓ |\n")
        md.append("|---------------|--------|-----------|-----------|-------------|\n")
        
        for att_type in attention_types:
            eval_results = results.get(f'{att_type}_eval', {})
            bleu = eval_results.get('bleu', 'N/A')
            rouge1 = eval_results.get('rouge1', 'N/A')
            rougeL = eval_results.get('rougeL', 'N/A')
            ppl = eval_results.get('perplexity', 'N/A')
            
            bleu_str = f"{bleu:.2f}" if isinstance(bleu, (int, float)) else bleu
            rouge1_str = f"{rouge1:.4f}" if isinstance(rouge1, (int, float)) else rouge1
            rougeL_str = f"{rougeL:.4f}" if isinstance(rougeL, (int, float)) else rougeL
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else ppl
            
            md.append(f"| **{att_type.upper()}** | {bleu_str} | {rouge1_str} | {rougeL_str} | {ppl_str} |\n")
        
        md.append("\n")
        
        # Best model
        valid_bleu = [(att, results[f'{att}_eval']['bleu']) 
                      for att in attention_types 
                      if f'{att}_eval' in results and results[f'{att}_eval'].get('bleu') is not None]
        
        if valid_bleu:
            best_bleu = max(valid_bleu, key=lambda x: x[1])
            md.append(f"**🏆 Best Translation Quality:** {best_bleu[0].upper()} ")
            md.append(f"(BLEU: {best_bleu[1]:.2f})\n\n")
        
        # Interpretation
        md.append("**Interpretation:**\n\n")
        md.append("- **BLEU Score:** Measures n-gram overlap with reference translation\n")
        md.append("  - 20-25: Good quality for this task\n")
        md.append("  - 25-30: Very good quality\n")
        md.append("  - 30+: Excellent quality\n")
        md.append("- **ROUGE-L:** Measures longest common subsequence (recall-oriented)\n")
        md.append("- **Perplexity:** Model confidence (lower is better)\n")
        md.append("  - < 10: Model is very confident\n")
        md.append("  - 10-20: Good confidence\n")
        md.append("  - > 20: Model struggles\n\n")
    else:
        md.append(" Evaluation results not found. Run `python evaluate.py --attention all` first.\n\n")
    
    md.append("---\n\n")
    
    # 3. Attention Analysis
    md.append("##  Attention Analysis\n\n")
    
    has_analysis = any(f'{att}_analysis' in results for att in attention_types)
    
    if has_analysis:
        md.append("### Attention Characteristics\n\n")
        md.append("| Attention Type | Mean Entropy | Mean Sharpness | Source Entropy |\n")
        md.append("|---------------|--------------|----------------|----------------|\n")
        
        for att_type in attention_types:
            analysis = results.get(f'{att_type}_analysis', {})
            entropy = analysis.get('entropy', {}).get('mean', 'N/A')
            sharpness = analysis.get('sharpness', {}).get('mean', 'N/A')
            src_entropy = analysis.get('source_entropy', {}).get('mean', 'N/A')
            
            entropy_str = f"{entropy:.4f}" if isinstance(entropy, (int, float)) else entropy
            sharpness_str = f"{sharpness:.4f}" if isinstance(sharpness, (int, float)) else sharpness
            src_entropy_str = f"{src_entropy:.4f}" if isinstance(src_entropy, (int, float)) else src_entropy
            
            md.append(f"| **{att_type.upper()}** | {entropy_str} | {sharpness_str} | {src_entropy_str} |\n")
        
        md.append("\n")
        
        # Best attention characteristics
        valid_entropy = [(att, results[f'{att}_analysis']['entropy']['mean'])
                        for att in attention_types
                        if f'{att}_analysis' in results]
        
        if valid_entropy:
            min_entropy = min(valid_entropy, key=lambda x: x[1])
            md.append(f"** Most Focused Attention:** {min_entropy[0].upper()} ")
            md.append(f"(Entropy: {min_entropy[1]:.4f})\n\n")
        
        # Interpretation
        md.append("**Interpretation:**\n\n")
        md.append("- **Entropy:** Measures attention uncertainty\n")
        md.append("  - < 1.0: Very sharp (focused on 1-2 words)\n")
        md.append("  - 1.0-1.5: Sharp (focused on few words)\n")
        md.append("  - 1.5-2.0: Moderate (distributed across several words)\n")
        md.append("  - > 2.0: Diffuse (spread across many words)\n")
        md.append("- **Sharpness:** Maximum attention weight\n")
        md.append("  - > 0.7: Very sharp attention\n")
        md.append("  - 0.5-0.7: Moderate sharpness\n")
        md.append("  - < 0.5: Diffuse attention\n")
        md.append("- **Source Entropy:** How many target words align to each source word\n\n")
    else:
        md.append(" Attention analysis not found. Run `python analyze_attention.py` first.\n\n")
    
    md.append("---\n\n")
    
    # 4. Training Progress
    md.append("## Training Progress\n\n")
    
    has_history = any(f'{att}_history' in results for att in attention_types)
    
    if has_history:
        md.append("### Final Training Metrics\n\n")
        md.append("| Attention Type | Final Train Loss | Final Val Loss | Final Val PPL | Epochs Trained |\n")
        md.append("|---------------|-----------------|----------------|---------------|----------------|\n")
        
        for att_type in attention_types:
            history = results.get(f'{att_type}_history', {})
            
            if history:
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                epochs = history.get('epochs', [])
                
                if train_losses and val_losses:
                    final_train_loss = train_losses[-1]
                    final_val_loss = val_losses[-1]
                    final_ppl = np.exp(final_val_loss)
                    num_epochs = len(epochs)
                    
                    md.append(f"| **{att_type.upper()}** | {final_train_loss:.4f} | ")
                    md.append(f"{final_val_loss:.4f} | {final_ppl:.2f} | {num_epochs} |\n")
        
        md.append("\n")
    
    md.append("---\n\n")
    
    # 5. Key Insights
    md.append("##  Key Insights\n\n")
    
    if has_eval and has_analysis:
        md.append("### Performance vs Attention Characteristics\n\n")
        
        # Collect data for insights
        insights_data = []
        for att_type in attention_types:
            if f'{att_type}_eval' in results and f'{att_type}_analysis' in results:
                eval_res = results[f'{att_type}_eval']
                analysis_res = results[f'{att_type}_analysis']
                
                insights_data.append({
                    'type': att_type,
                    'bleu': eval_res.get('bleu'),
                    'ppl': eval_res.get('perplexity'),
                    'entropy': analysis_res['entropy']['mean'],
                    'sharpness': analysis_res['sharpness']['mean']
                })
        
        if insights_data:
            # Find correlations
            md.append("1. **Attention Sharpness:** ")
            
            # Sort by sharpness
            sorted_by_sharpness = sorted(insights_data, key=lambda x: x['sharpness'], reverse=True)
            md.append(f"{sorted_by_sharpness[0]['type'].upper()} has the sharpest attention ")
            md.append(f"(sharpness: {sorted_by_sharpness[0]['sharpness']:.4f}), ")
            
            # Check if it correlates with BLEU
            if sorted_by_sharpness[0]['bleu'] is not None:
                sorted_by_bleu = sorted([x for x in insights_data if x['bleu'] is not None], 
                                       key=lambda x: x['bleu'], reverse=True)
                if sorted_by_sharpness[0]['type'] == sorted_by_bleu[0]['type']:
                    md.append("which also achieves the highest BLEU score. ")
                    md.append("**This suggests sharper attention leads to better translation quality.**\n\n")
                else:
                    md.append("but not the highest BLEU score. ")
                    md.append("**Attention sharpness alone doesn't determine translation quality.**\n\n")
            
            # Entropy insight
            md.append("2. **Attention Entropy:** ")
            sorted_by_entropy = sorted(insights_data, key=lambda x: x['entropy'])
            md.append(f"{sorted_by_entropy[0]['type'].upper()} has the lowest entropy ")
            md.append(f"({sorted_by_entropy[0]['entropy']:.4f}), indicating the most focused attention. ")
            
            if sorted_by_entropy[0]['ppl'] is not None:
                sorted_by_ppl = sorted([x for x in insights_data if x['ppl'] is not None],
                                      key=lambda x: x['ppl'])
                if sorted_by_entropy[0]['type'] == sorted_by_ppl[0]['type']:
                    md.append("It also has the lowest perplexity, showing higher model confidence.\n\n")
                else:
                    md.append("\n\n")
            
            # Overall winner
            md.append("3. **Overall Best Model:** ")
            if sorted_by_bleu:
                best = sorted_by_bleu[0]
                md.append(f"**{best['type'].upper()}** achieves the best balance of ")
                md.append(f"performance (BLEU: {best['bleu']:.2f}) and ")
                md.append(f"attention quality (Entropy: {best['entropy']:.4f}).\n\n")
    
    # General insights
    md.append("### General Observations\n\n")
    md.append("1. **Bahdanau (Additive) Attention:**\n")
    md.append("   - Most expressive (many learnable parameters)\n")
    md.append("   - Often produces sharpest alignments\n")
    md.append("   - Slower training and inference\n\n")
    
    md.append("2. **Luong (Multiplicative) Attention:**\n")
    md.append("   - Simpler formulation\n")
    md.append("   - Faster computation\n")
    md.append("   - Competitive performance with fewer parameters\n\n")
    
    md.append("3. **Scaled Dot-Product Attention:**\n")
    md.append("   - Most stable training (scaling factor prevents gradient issues)\n")
    md.append("   - Used in modern Transformers\n")
    md.append("   - Good balance of performance and efficiency\n\n")
    
    md.append("---\n\n")
    
    # 6. Sample Translations
    md.append("##  Sample Translations\n\n")
    
    for att_type in attention_types:
        eval_results = results.get(f'{att_type}_eval', {})
        samples = eval_results.get('sample_translations', [])
        
        if samples:
            md.append(f"### {att_type.upper()} Attention\n\n")
            
            for i, sample in enumerate(samples[:3], 1):  # Show first 3
                md.append(f"**Example {i}:**\n")
                md.append(f"- **Reference:** {sample['reference']}\n")
                md.append(f"- **Translation:** {sample['hypothesis']}\n\n")
    
    md.append("---\n\n")
    
    # 7. Recommendations
    md.append("##  Recommendations\n\n")
    md.append("### When to Use Each Attention Mechanism\n\n")
    md.append("1. **Use Bahdanau Attention when:**\n")
    md.append("   - You need maximum translation quality\n")
    md.append("   - Training time is not a constraint\n")
    md.append("   - Complex alignment patterns are expected\n\n")
    
    md.append("2. **Use Luong Attention when:**\n")
    md.append("   - You need faster training/inference\n")
    md.append("   - Memory is limited (fewer parameters)\n")
    md.append("   - Performance vs efficiency trade-off is important\n\n")
    
    md.append("3. **Use Scaled Dot-Product Attention when:**\n")
    md.append("   - Building modern architectures (Transformers)\n")
    md.append("   - Training stability is crucial\n")
    md.append("   - You want state-of-the-art performance\n\n")
    
    md.append("---\n\n")
    
    # 8. Reproducibility
    md.append("##  Reproducibility\n\n")
    md.append("All experiments were conducted with:\n\n")
    md.append("- **Random Seed:** 42 (Python, NumPy, PyTorch)\n")
    md.append("- **Dataset:** Multi30k (English-German)\n")
    md.append("- **Architecture:** Bidirectional GRU Encoder + GRU Decoder\n")
    md.append("- **Hyperparameters:** (see `config.yaml`)\n")
    md.append("  - Embedding dim: 256\n")
    md.append("  - Hidden dim: 512\n")
    md.append("  - Layers: 2\n")
    md.append("  - Dropout: 0.3\n")
    md.append("  - Batch size: 128\n")
    md.append("  - Learning rate: 0.001\n\n")
    
    md.append("---\n\n")
    
    # 9. Visualizations
    md.append("##  Visualizations\n\n")
    md.append("Attention heatmaps and analysis plots are available in:\n")
    md.append("- `experiments/<attention_type>/visualizations/` - Individual attention maps\n")
    md.append("- `experiments/visualizations/` - Comparison plots\n\n")
    
    md.append("---\n\n")
    
    # Footer
    md.append("##  References\n\n")
    md.append("1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). ")
    md.append("*Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015.\n\n")
    md.append("2. Luong, M. T., Pham, H., & Manning, C. D. (2015). ")
    md.append("*Effective Approaches to Attention-based Neural Machine Translation*. EMNLP 2015.\n\n")
    md.append("3. Vaswani, A., et al. (2017). ")
    md.append("*Attention Is All You Need*. NeurIPS 2017.\n\n")
    
    md.append("---\n\n")
    md.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(md)
    
    print(f"[DONE] Markdown report saved: {output_path}")


def main():
    """Main report generation function."""
    
    print("\n" + "=" * 70)
    print("[INFO] GENERATING FINAL REPORT")
    print("=" * 70)
    
    # Load all results
    print("\n[INFO] Loading results...")
    results = load_results()
    
    loaded_count = len([k for k in results.keys()])
    print(f"   [OK] Loaded {loaded_count} result files")
    
    # Generate report
    print("\n[INFO] Generating report...")
    output_path = Path("experiments/FINAL_REPORT.md")
    generate_markdown_report(results, output_path)
    
    print("\n" + "=" * 70)
    print("[DONE] REPORT GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n[INFO] Report: {output_path}")
    print(f"\nYou can read the report with:")
    print(f"   - Any text editor")
    print(f"   - Markdown viewer")
    print(f"   - GitHub (renders markdown automatically)")


if __name__ == "__main__":
    main()

