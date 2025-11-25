import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from pathlib import Path
import torch

class LIMEAnalyzer:
        def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config['output']['visualizations_dir']) / 'lime'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(
            class_names=['Negative', 'Positive'],
            bow=False  # Don't use bag-of-words, preserve word order
        )
    
    def predict_proba(self, texts):
                probabilities = []
        
        for text in texts:
            # Tokenize
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
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)
    
    def explain_instance(self, text, num_features=10, num_samples=1000):
                # Generate explanation
        exp = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction
        probs = self.predict_proba([text])[0]
        predicted_class = int(np.argmax(probs))
        confidence = probs[predicted_class]
        
        # Extract feature weights
        feature_weights = exp.as_list()
        
        return {
            'explanation': exp,
            'feature_weights': feature_weights,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs,
            'text': text
        }
    
    def visualize_explanation(self, explanation_data, sample_name, save=True):
                feature_weights = explanation_data['feature_weights']
        predicted_class = explanation_data['predicted_class']
        confidence = explanation_data['confidence']
        text = explanation_data['text']
        
        # Separate features by class
        features = [fw[0] for fw in feature_weights]
        weights = [fw[1] for fw in feature_weights]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Feature importance bar chart
        colors = ['red' if w < 0 else 'green' for w in weights]
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, weights, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Importance Weight')
        ax1.set_title(
            f'LIME Feature Importance\n'
            f'Predicted: {"Positive" if predicted_class == 1 else "Negative"} '
            f'(Confidence: {confidence:.3f})'
        )
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Text with highlighted features
        ax2.axis('off')
        
        # Create highlighted text
        ax2.text(0.05, 0.95, 'Original Text with Feature Highlighting:',
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top')
        
        # Highlight important words in text
        words = text.split()
        feature_dict = dict(feature_weights)
        
        y_offset = 0.85
        x_offset = 0.05
        max_width = 0.9
        
        for word in words:
            # Check if word (or cleaned version) is in features
            cleaned_word = word.lower().strip('.,!?;:()')
            weight = feature_dict.get(cleaned_word, 0)
            
            # Color based on weight
            if abs(weight) > 0.01:
                if weight > 0:
                    # Green for positive (supports positive class)
                    intensity = min(abs(weight) * 2, 1.0)
                    color = (1 - intensity, 1, 1 - intensity)
                else:
                    # Red for negative (supports negative class)
                    intensity = min(abs(weight) * 2, 1.0)
                    color = (1, 1 - intensity, 1 - intensity)
                
                bbox_props = dict(boxstyle='round', facecolor=color, alpha=0.7)
            else:
                bbox_props = None
            
            # Add word
            txt = ax2.text(x_offset, y_offset, word + ' ',
                          transform=ax2.transAxes, fontsize=9,
                          bbox=bbox_props, verticalalignment='top')
            
            # Update position
            bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax2.transData.inverted())
            word_width = bbox_data.width / fig.get_figwidth() / 2
            
            x_offset += word_width
            if x_offset > max_width:
                x_offset = 0.05
                y_offset -= 0.05
        
        # Add legend
        legend_y = 0.15
        ax2.text(0.05, legend_y, 'Legend:',
                transform=ax2.transAxes, fontsize=10, fontweight='bold')
        ax2.text(0.05, legend_y - 0.05, '● Green = Supports Positive sentiment',
                transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=(0.7, 1, 0.7), alpha=0.7))
        ax2.text(0.05, legend_y - 0.10, '● Red = Supports Negative sentiment',
                transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=(1, 0.7, 0.7), alpha=0.7))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{sample_name}_lime_explanation.png'
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def visualize_probability_distribution(self, explanation_data, sample_name):
                probs = explanation_data['probabilities']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        classes = ['Negative', 'Positive']
        colors = ['red', 'green']
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability')
        ax.set_title(
            f'Prediction Probability Distribution\n'
            f'Text: {explanation_data["text"][:60]}...'
        )
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{sample_name}_probability.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                   bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def compare_explanations(self, texts, sample_names):
                explanations = []
        all_features = []
        
        for text in texts:
            exp_data = self.explain_instance(text)
            explanations.append(exp_data)
            all_features.extend([fw[0] for fw in exp_data['feature_weights']])
        
        # Find most common features
        from collections import Counter
        feature_counts = Counter(all_features)
        top_features = [f for f, _ in feature_counts.most_common(15)]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(top_features))
        width = 0.8 / len(texts)
        
        for idx, (exp_data, name) in enumerate(zip(explanations, sample_names)):
            feature_dict = dict(exp_data['feature_weights'])
            weights = [feature_dict.get(f, 0) for f in top_features]
            
            offset = (idx - len(texts)/2) * width + width/2
            ax.bar(x + offset, weights, width, label=name, alpha=0.7)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Weight')
        ax.set_title('LIME Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'lime_comparison.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                   bbox_inches='tight')
        plt.close()
        
        return str(save_path)
