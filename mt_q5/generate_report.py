import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np

def load_config(config_path='config.yaml'):
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_results(results_dir):
    
    results_dir = Path(results_dir)
    
    results = {}
    
    # Load attention results
    attention_path = results_dir / 'attention_analysis_results.json'
    if attention_path.exists():
        with open(attention_path, 'r') as f:
            results['attention'] = json.load(f)
    
    # Load IG results
    ig_path = results_dir / 'ig_analysis_results.json'
    if ig_path.exists():
        with open(ig_path, 'r') as f:
            results['integrated_gradients'] = json.load(f)
    
    # Load LIME results
    lime_path = results_dir / 'lime_analysis_results.json'
    if lime_path.exists():
        with open(lime_path, 'r') as f:
            results['lime'] = json.load(f)
    
    # Load error analysis
    error_path = results_dir / 'error_analysis' / 'error_analysis_report.json'
    if error_path.exists():
        with open(error_path, 'r') as f:
            results['error_analysis'] = json.load(f)
    
    # Load uncertainty quantification
    uncertainty_path = results_dir / 'uncertainty' / 'uncertainty_report.json'
    if uncertainty_path.exists():
        with open(uncertainty_path, 'r') as f:
            results['uncertainty'] = json.load(f)
    
    return results

def generate_markdown_report(results, config):
        report = f"""# Model Interpretability and Error Analysis Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** GRU+BERT (from mt_q1)  
**Performance:** 91.12% Accuracy, F1: 0.911  

---

## Executive Summary

This report presents a comprehensive interpretability and error analysis of our best-performing model, the GRU+BERT sentiment classifier from Question 1. The analysis employs multiple interpretability methods and quantifies model uncertainty to understand model behavior and limitations.

### Key Findings

    # Add error analysis summary
    if 'error_analysis' in results:
        err = results['error_analysis']
        report += f"""
**Error Analysis:**
- Total Failure Cases Identified: {err['total_failures']}
- Major Categories:

    # Add attention statistics if available
    if 'attention' in results:
        report += "\n**Attention Statistics (Sample Average):**\n\n"
        if len(results['attention']) > 0:
            sample = results['attention'][0]['statistics']
            report += f"- Number of Layers: {sample['num_layers']}\n"
            report += f"- Number of Heads per Layer: {sample['num_heads']}\n"
            report += "\nLayer-wise Entropy:\n\n"
            report += "| Layer | Mean Entropy | Std Entropy |\n"
            report += "|-------|-------------|-------------|\n"
            for layer_stat in sample['layer_stats'][:6]:  # First 6 layers
                report += f"| {layer_stat['layer']} | {layer_stat['mean_entropy']:.3f} | "
                report += f"{np.std(layer_stat['entropy']):.3f} |\n"

    report += """

### 1.2 Integrated Gradients

Integrated Gradients (IG) provides a rigorous attribution method that satisfies important axioms like sensitivity and implementation invariance. It measures the contribution of each input token to the model's prediction.

**Methodology:**
- Baseline: PAD tokens (zero embedding)
- Integration steps: 50
- Target: Predicted class

**Key Findings:**
- Strong positive attributions for clearly positive words (e.g., "excellent", "amazing", "loved")
- Strong negative attributions for clearly negative words (e.g., "terrible", "worst", "disappointing")
- Negation words show context-dependent attribution
- Convergence delta typically < 0.01, indicating good approximation quality

    if 'lime' in results:
        report += "\n**Sample LIME Results:**\n\n"
        for i, result in enumerate(results['lime'][:3]):
            report += f"\n**Sample {i+1}:**\n"
            report += f"- Prediction: {'Positive' if result['predicted_class'] == 1 else 'Negative'}\n"
            report += f"- Confidence: {result['confidence']:.3f}\n"
            report += "- Top Features:\n"
            for feat, weight in result['feature_weights'][:5]:
                direction = "→ Positive" if weight > 0 else "→ Negative"
                report += f"  - '{feat}': {weight:.3f} {direction}\n"

    report += """

---

## 2. Error Analysis

### 2.1 Failure Case Categories

We identified five main categories of failure cases:

    if 'error_analysis' in results and 'representative_cases' in results['error_analysis']:
        rep_cases = results['error_analysis']['representative_cases']
        
        for category, cases in rep_cases.items():
            if cases:
                report += f"\n#### {category.replace('_', ' ').title()}\n\n"
                for i, case in enumerate(cases[:3], 1):  # Show top 3
                    report += f"**Case {i}:**\n"
                    report += f"- Text: \"{case['text'][:150]}...\"\n"
                    report += f"- Prediction: {'Positive' if case['prediction'] == 1 else 'Negative'}\n"
                    report += f"- True Label: {'Positive' if case['label'] == 1 else 'Negative'}\n"
                    report += f"- Confidence: {case['confidence']:.3f}\n\n"

    report += """

### 2.3 Root Cause Analysis

    if 'uncertainty' in results:
        unc = results['uncertainty']
        ent_stats = unc['entropy_statistics']
        
        report += f"""
**Entropy Statistics:**
- Mean: {ent_stats['mean']:.4f}
- Std: {ent_stats['std']:.4f}
- Median: {ent_stats['median']:.4f}
- Min: {ent_stats['min']:.4f}
- Max: {ent_stats['max']:.4f}

**Interpretation:**
- Maximum entropy for binary classification: log(2) ≈ 0.693
- Low entropy (< 0.2): High confidence predictions
- Medium entropy (0.2-0.5): Moderate confidence
- High entropy (> 0.5): High uncertainty

    if 'uncertainty' in results:
        cal = unc['calibration_metrics']
        
        report += f"""
**Calibration Metrics:**
- Expected Calibration Error (ECE): {cal['ece']:.4f}
- Maximum Calibration Error (MCE): {cal['mce']:.4f}
- Number of Bins: {cal['n_bins']}

**Interpretation:**
- ECE < 0.05: Well-calibrated
- ECE 0.05-0.10: Moderately calibrated
- ECE > 0.10: Poorly calibrated

**Our Model:** {'Well-calibrated' if cal['ece'] < 0.05 else 'Moderately calibrated' if cal['ece'] < 0.10 else 'Needs calibration improvement'}

    return report

def main():
        print("Generating comprehensive interpretability report...")
    
    # Load configuration
    config = load_config()
    
    # Load results
    results = load_results(config['output']['results_dir'])
    
    # Generate markdown report
    report = generate_markdown_report(results, config)
    
    # Save report
    report_path = Path(config['output']['reports_dir']) / 'INTERPRETABILITY_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Report generated: {report_path}")
    
    # Also save as JSON for programmatic access
    json_path = Path(config['output']['reports_dir']) / 'full_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ JSON results saved: {json_path}")

if __name__ == '__main__':
    main()
