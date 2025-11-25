# Error analysis module for identifying and analyzing model failure cases

import numpy as np
import torch
from pathlib import Path
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Convert numpy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

class ErrorAnalyzer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config['output']['results_dir']) / 'error_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vocab_freq = self._load_vocabulary_stats()
        
    def _load_vocabulary_stats(self):
        # Load vocabulary frequency statistics from training data
        vocab_path = Path(self.config['model']['checkpoint_path']).parent / 'vocabulary.json'
        
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                return vocab_data.get('word_freq', {})
        return {}
    
    def predict_with_confidence(self, text):
        # Make prediction and return detailed information
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
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'tokens': tokens,
            'text': text
        }
    
    # Identify different types of failure cases
    def identify_failure_cases(self, dataset, predictions, labels):
        failure_cases = {
            'low_confidence_correct': [],
            'high_confidence_wrong': [],
            'rare_words': [],
            'long_text': [],
            'short_text': [],
            'ambiguous_context': []
        }
        
        for idx, (text, pred, label) in enumerate(zip(dataset, predictions, labels)):
            pred_data = self.predict_with_confidence(text)
            confidence = pred_data['confidence']
            prediction = pred_data['prediction']
            
            if prediction == label and confidence < 0.6:
                failure_cases['low_confidence_correct'].append({
                    'index': idx,
                    'text': text,
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence
                })
            
            if prediction != label and confidence > 0.7:
                failure_cases['high_confidence_wrong'].append({
                    'index': idx,
                    'text': text,
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence
                })
            
            if self._contains_rare_words(text):
                failure_cases['rare_words'].append({
                    'index': idx,
                    'text': text,
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence,
                    'rare_words': self._get_rare_words(text)
                })
            
            word_count = len(text.split())
            if word_count > 300:
                failure_cases['long_text'].append({
                    'index': idx,
                    'text': text[:200] + '...',
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence,
                    'word_count': word_count
                })
            elif word_count < 20:
                failure_cases['short_text'].append({
                    'index': idx,
                    'text': text,
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence,
                    'word_count': word_count
                })
            
            if 0.4 < confidence < 0.6:
                failure_cases['ambiguous_context'].append({
                    'index': idx,
                    'text': text,
                    'prediction': prediction,
                    'label': label,
                    'confidence': confidence
                })
        
        return failure_cases
    
    def _contains_rare_words(self, text, threshold=10):
        # Check if text contains rare words based on vocabulary frequency
        if not self.vocab_freq:
            return False
        
        words = text.lower().split()
        for word in words:
            freq = self.vocab_freq.get(word, 0)
            if 0 < freq < threshold:
                return True
        return False
    
    def _get_rare_words(self, text, threshold=10):
        # Get list of rare words in text
        if not self.vocab_freq:
            return []
        
        words = text.lower().split()
        rare_words = []
        for word in words:
            freq = self.vocab_freq.get(word, 0)
            if 0 < freq < threshold:
                rare_words.append((word, freq))
        return rare_words
    
    def analyze_failure_patterns(self, failure_cases):
        # Analyze patterns in failure cases
        analysis = {
            'summary': {},
            'patterns': {}
        }
        
        for category, cases in failure_cases.items():
            analysis['summary'][category] = {
                'count': len(cases),
                'avg_confidence': np.mean([c['confidence'] for c in cases]) if cases else 0
            }
        
        if failure_cases['high_confidence_wrong']:
            all_words = []
            for case in failure_cases['high_confidence_wrong']:
                all_words.extend(case['text'].lower().split())
            
            word_freq = Counter(all_words)
            analysis['patterns']['high_conf_wrong_common_words'] = word_freq.most_common(20)
        
        if failure_cases['rare_words']:
            rare_word_list = []
            for case in failure_cases['rare_words']:
                rare_word_list.extend([w[0] for w in case.get('rare_words', [])])
            
            rare_freq = Counter(rare_word_list)
            analysis['patterns']['most_problematic_rare_words'] = rare_freq.most_common(15)
        
        if failure_cases['long_text'] or failure_cases['short_text']:
            long_counts = [c['word_count'] for c in failure_cases['long_text']]
            short_counts = [c['word_count'] for c in failure_cases['short_text']]
            
            analysis['patterns']['length_distribution'] = {
                'long_avg': np.mean(long_counts) if long_counts else 0,
                'short_avg': np.mean(short_counts) if short_counts else 0
            }
        
        return analysis
    
    # Select representative failure cases for detailed analysis
    def select_representative_failures(self, failure_cases, num_per_category=5):
        representatives = {}
        
        for category, cases in failure_cases.items():
            if not cases:
                representatives[category] = []
                continue
            
            if category == 'high_confidence_wrong':
                sorted_cases = sorted(cases, key=lambda x: x['confidence'], reverse=True)
            elif category == 'low_confidence_correct':
                sorted_cases = sorted(cases, key=lambda x: x['confidence'])
            elif category == 'ambiguous_context':
                sorted_cases = sorted(cases, key=lambda x: abs(x['confidence'] - 0.5))
            else:
                sorted_cases = cases
            
            representatives[category] = sorted_cases[:num_per_category]
        
        return representatives
    
    def visualize_failure_distribution(self, failure_cases, save_path=None):
        # Visualize distribution of failure cases across categories
        categories = list(failure_cases.keys())
        counts = [len(cases) for cases in failure_cases.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.Set3(range(len(categories)))
        bars = ax1.bar(range(len(categories)), counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0, ha='center')
        ax1.set_ylabel('Number of Cases')
        ax1.set_title('Failure Case Distribution by Category')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.pie(counts, labels=[cat.replace('_', ' ').title() for cat in categories],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Failure Case Proportion')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'failure_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_confidence_distribution(self, failure_cases, save_path=None):
        # Visualize confidence distribution for different failure types
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (category, cases) in enumerate(failure_cases.items()):
            if idx >= len(axes):
                break
            
            if cases:
                confidences = [c['confidence'] for c in cases]
                
                axes[idx].hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel('Confidence')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{category.replace("_", " ").title()}\n(n={len(cases)})')
                axes[idx].axvline(x=np.mean(confidences), color='red', linestyle='--',
                                 label=f'Mean: {np.mean(confidences):.3f}')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, 'No cases', ha='center', va='center',
                              transform=axes[idx].transAxes, fontsize=12)
                axes[idx].set_title(category.replace('_', ' ').title())
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'confidence_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_error_report(self, failure_cases, analysis, representative_cases):
        # Generate comprehensive error analysis report
        report = {
            'summary': analysis['summary'],
            'patterns': analysis['patterns'],
            'total_failures': sum(len(cases) for cases in failure_cases.values()),
            'representative_cases': representative_cases
        }
        
        report = convert_numpy_types(report)
        
        report_path = self.output_dir / 'error_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
