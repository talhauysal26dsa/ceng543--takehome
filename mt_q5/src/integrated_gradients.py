import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from pathlib import Path

class IntegratedGradientsAnalyzer:
        def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config['output']['visualizations_dir']) / 'integrated_gradients'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use LayerIntegratedGradients on the embedding layer
        # This handles the embedding layer properly without interpolation issues
        self.embeddings = self.model.bert.embeddings.word_embeddings
        self.ig = LayerIntegratedGradients(self.forward_func, self.embeddings)
        
    def forward_func(self, input_ids, attention_mask=None):
                if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def compute_attributions(self, text, target_class=None, n_steps=50):
                # Tokenize input
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
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        # Create baseline (all PAD tokens)
        baseline_ids = torch.ones_like(input_ids) * self.tokenizer.pad_token_id
        
        # Compute attributions using LayerIntegratedGradients
        # This operates on the input_ids directly and handles the embedding layer internally
        attributions, delta = self.ig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=target_class,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # LayerIG returns attributions at embedding layer - sum across embedding dimension
        attributions = attributions.sum(dim=-1)
        
        # Convert to numpy
        attributions = attributions.cpu().detach().numpy()[0]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out padding
        valid_length = attention_mask.sum().item()
        attributions = attributions[:valid_length]
        tokens = tokens[:valid_length]
        
        return {
            'attributions': attributions,
            'tokens': tokens,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence,
            'convergence_delta': delta.item(),
            'text': text
        }
    
    def visualize_attributions(self, attribution_data, sample_name, save=True):
                tokens = attribution_data['tokens']
        attributions = attribution_data['attributions']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Bar chart of attributions
        x_pos = np.arange(len(tokens))
        colors = ['red' if attr < 0 else 'green' for attr in attributions]
        
        ax1.bar(x_pos, attributions, color=colors, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tokens, rotation=90, ha='right', fontsize=8)
        ax1.set_ylabel('Attribution Score')
        ax1.set_title(
            f'Integrated Gradients Attribution\n'
            f'Predicted: {attribution_data["predicted_class"]} '
            f'(Confidence: {attribution_data["confidence"]:.3f})'
        )
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Text visualization with color coding
        ax2.axis('off')
        
        # Normalize attributions for color mapping
        abs_max = max(abs(attributions.min()), abs(attributions.max()))
        if abs_max > 0:
            norm_attr = attributions / abs_max
        else:
            norm_attr = attributions
        
        # Create colored text
        text_viz = "Token Attribution (Green=Positive, Red=Negative):\n\n"
        for token, attr, norm_a in zip(tokens, attributions, norm_attr):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Color intensity based on attribution magnitude
            if norm_a > 0:
                color = (1 - norm_a, 1, 1 - norm_a)  # Green gradient
            else:
                color = (1, 1 + norm_a, 1 + norm_a)  # Red gradient
            
            # Display token with background color
            ax2.text(
                0.05, 0.9 - len(text_viz.split('\n')) * 0.03,
                f'{token} ({attr:.3f})',
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                fontsize=9
            )
            text_viz += f'{token} '
        
        info_text = (
            f"Original Text:\n{attribution_data['text'][:200]}...\n\n"
            f"Convergence Delta: {attribution_data['convergence_delta']:.6f}\n"
            f"(Lower delta indicates better approximation)"
        )
        ax2.text(0.05, 0.3, info_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'{sample_name}_ig_attribution.png'
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def visualize_top_attributions(self, attribution_data, sample_name, top_k=10):
                tokens = attribution_data['tokens']
        attributions = attribution_data['attributions']
        
        # Get top-k by absolute value
        abs_attributions = np.abs(attributions)
        top_indices = np.argsort(abs_attributions)[-top_k:][::-1]
        
        top_tokens = [tokens[i] for i in top_indices]
        top_attrs = [attributions[i] for i in top_indices]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        colors = ['red' if attr < 0 else 'green' for attr in top_attrs]
        
        plt.barh(range(top_k), top_attrs, color=colors, alpha=0.7)
        plt.yticks(range(top_k), top_tokens)
        plt.xlabel('Attribution Score')
        plt.title(f'Top {top_k} Most Important Tokens\n'
                 f'Class: {attribution_data["predicted_class"]}, '
                 f'Confidence: {attribution_data["confidence"]:.3f}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / f'{sample_name}_top{top_k}_tokens.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight')
        plt.close()
        
        return {
            'top_tokens': top_tokens,
            'top_attributions': top_attrs,
            'save_path': str(save_path)
        }
    
    def compare_attributions(self, texts, sample_names, target_class=None):
                all_attributions = []
        all_tokens_list = []
        
        for text in texts:
            attr_data = self.compute_attributions(text, target_class)
            all_attributions.append(attr_data['attributions'])
            all_tokens_list.append(attr_data['tokens'])
        
        # Create comparison visualization
        fig, axes = plt.subplots(len(texts), 1, figsize=(16, 4 * len(texts)))
        if len(texts) == 1:
            axes = [axes]
        
        for idx, (tokens, attributions, name) in enumerate(
            zip(all_tokens_list, all_attributions, sample_names)
        ):
            x_pos = np.arange(len(tokens))
            colors = ['red' if attr < 0 else 'green' for attr in attributions]
            
            axes[idx].bar(x_pos, attributions, color=colors, alpha=0.7)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(tokens, rotation=90, ha='right', fontsize=7)
            axes[idx].set_ylabel('Attribution')
            axes[idx].set_title(f'{name}')
            axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'attribution_comparison.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight')
        plt.close()
        
        return str(save_path)
