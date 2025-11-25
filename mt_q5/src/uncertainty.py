import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import softmax
from sklearn.calibration import calibration_curve
import json

class UncertaintyQuantifier:
        def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config['output']['results_dir']) / 'uncertainty'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_entropy(self, probabilities):
                # Add small epsilon to avoid log(0)
        eps = 1e-10
        probabilities = np.clip(probabilities, eps, 1.0)
        
        entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
        return entropy
    
    def compute_predictive_entropy(self, dataset, batch_size=32):
                entropies = []
        predictions = []
        confidences = []
        
        self.model.eval()
        
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i:i + batch_size]
            
            for text in batch_texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    
                    entropy = self.compute_entropy(probs)
                    prediction = np.argmax(probs)
                    confidence = probs[prediction]
                    
                    entropies.append(entropy)
                    predictions.append(prediction)
                    confidences.append(confidence)
        
        return {
            'entropies': np.array(entropies),
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'median_entropy': np.median(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies)
        }
    
    def compute_calibration(self, probabilities, predictions, labels, n_bins=10):
                # Get confidence for predicted class
        confidences = probabilities[np.arange(len(predictions)), predictions]
        
        # Compute accuracy
        accuracy = (predictions == labels).astype(float)
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            accuracy,
            confidences,
            n_bins=n_bins,
            strategy='uniform'
        )
        
        # Compute Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences, accuracy, n_bins)
        
        # Compute Maximum Calibration Error (MCE)
        mce = np.max(np.abs(prob_true - prob_pred))
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'mce': mce,
            'n_bins': n_bins
        }
    
    def _compute_ece(self, confidences, accuracies, n_bins=10):
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    def mc_dropout_predict(self, text, num_samples=30, dropout_rate=0.2):
                # Prepare input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Enable dropout
        self.model.train()
        
        # Collect predictions
        all_predictions = []
        all_probabilities = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
                
                all_predictions.append(pred)
                all_probabilities.append(probs)
        
        # Reset to eval mode
        self.model.eval()
        
        all_probabilities = np.array(all_probabilities)
        
        # Compute statistics
        mean_probs = np.mean(all_probabilities, axis=0)
        std_probs = np.std(all_probabilities, axis=0)
        
        # Predictive entropy (uncertainty in prediction)
        predictive_entropy = self.compute_entropy(mean_probs)
        
        # Mutual information (epistemic uncertainty)
        expected_entropy = np.mean([self.compute_entropy(p) for p in all_probabilities])
        mutual_info = predictive_entropy - expected_entropy
        
        return {
            'mean_probabilities': mean_probs,
            'std_probabilities': std_probs,
            'predictions': all_predictions,
            'prediction_counts': np.bincount(all_predictions, minlength=2),
            'prediction_entropy': predictive_entropy,
            'mutual_information': mutual_info,
            'text': text
        }
    
    def visualize_entropy_distribution(self, entropy_data, save_path=None):
                entropies = entropy_data['entropies']
        confidences = entropy_data['confidences']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram of entropies
        axes[0, 0].hist(entropies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(entropy_data['mean_entropy'], color='red', linestyle='--',
                          label=f'Mean: {entropy_data["mean_entropy"]:.3f}')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Entropy Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Scatter plot: Entropy vs Confidence
        axes[0, 1].scatter(confidences, entropies, alpha=0.3, s=10)
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].set_title('Entropy vs Confidence')
        axes[0, 1].grid(alpha=0.3)
        
        # Box plot by confidence bins
        conf_bins = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        binned_entropies = []
        bin_labels = []
        
        for i, label in enumerate(conf_bins, start=5):
            bin_data = entropies[(confidences >= i/10) & (confidences < (i+1)/10)]
            if len(bin_data) > 0:  # Only include non-empty bins
                binned_entropies.append(bin_data)
                bin_labels.append(label)
        
        if binned_entropies:  # Only create boxplot if we have data
            axes[1, 0].boxplot(binned_entropies, labels=bin_labels)
            axes[1, 0].set_xlabel('Confidence Range')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].set_title('Entropy by Confidence Bins')
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[1, 0].set_title('Entropy by Confidence Bins (No Data)')
        
        # CDF of entropy
        sorted_entropies = np.sort(entropies)
        cdf = np.arange(1, len(sorted_entropies) + 1) / len(sorted_entropies)
        axes[1, 1].plot(sorted_entropies, cdf, linewidth=2)
        axes[1, 1].set_xlabel('Entropy')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Entropy CDF')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'entropy_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_calibration(self, calibration_data, save_path=None):
                prob_true = calibration_data['prob_true']
        prob_pred = calibration_data['prob_pred']
        ece = calibration_data['ece']
        mce = calibration_data['mce']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                label=f'Model (ECE={ece:.4f})')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve (Reliability Diagram)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Bar chart showing calibration gap
        gaps = np.abs(prob_true - prob_pred)
        bins = [f'Bin {i+1}' for i in range(len(gaps))]
        
        colors = ['red' if gap > 0.05 else 'green' for gap in gaps]
        ax2.bar(range(len(gaps)), gaps, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(gaps)))
        ax2.set_xticklabels(bins, rotation=45)
        ax2.set_ylabel('Calibration Gap')
        ax2.set_title(f'Calibration Gap per Bin\nMCE (Max Gap): {mce:.4f}')
        ax2.axhline(y=0.05, color='orange', linestyle='--', label='5% threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'calibration_curve.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_mc_dropout_uncertainty(self, mc_data, save_path=None):
                mean_probs = mc_data['mean_probabilities']
        std_probs = mc_data['std_probabilities']
        pred_counts = mc_data['prediction_counts']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean probabilities with error bars
        classes = ['Negative', 'Positive']
        x_pos = np.arange(len(classes))
        
        ax1.bar(x_pos, mean_probs, yerr=std_probs, capsize=10,
               color=['red', 'green'], alpha=0.7, edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(classes)
        ax1.set_ylabel('Probability')
        ax1.set_title(f'MC Dropout Predictions (n={len(mc_data["predictions"])} samples)\n'
                     f'Predictive Entropy: {mc_data["prediction_entropy"]:.4f}')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Prediction distribution
        ax2.bar(x_pos, pred_counts, color=['red', 'green'], alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(classes)
        ax2.set_ylabel('Count')
        ax2.set_title(f'Prediction Distribution\n'
                     f'Mutual Information: {mc_data["mutual_information"]:.4f}')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'mc_dropout_uncertainty.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_uncertainty_report(self, entropy_data, calibration_data):
                report = {
            'entropy_statistics': {
                'mean': float(entropy_data['mean_entropy']),
                'std': float(entropy_data['std_entropy']),
                'median': float(entropy_data['median_entropy']),
                'min': float(entropy_data['min_entropy']),
                'max': float(entropy_data['max_entropy'])
            },
            'calibration_metrics': {
                'ece': float(calibration_data['ece']),
                'mce': float(calibration_data['mce']),
                'n_bins': calibration_data['n_bins']
            }
        }
        
        # Save to JSON
        report_path = self.output_dir / 'uncertainty_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
