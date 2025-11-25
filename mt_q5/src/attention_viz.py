import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

class AttentionVisualizer:
        def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output']['visualizations_dir']) / 'attention'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_attention_heatmap(self, attention_weights, tokens, layer_idx, head_idx, 
                                    sample_name, title=None):
                plt.figure(figsize=(12, 10))
        
        # Filter out padding tokens for cleaner visualization
        valid_length = len([t for t in tokens if t != '[PAD]'])
        attention_weights = attention_weights[:valid_length, :valid_length]
        tokens = tokens[:valid_length]
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            cbar=True,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f'{sample_name}_layer{layer_idx}_head{head_idx}.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_average_attention(self, attention_data, sample_name, text=None):
                attentions = attention_data['attentions']
        tokens = attention_data['tokens']
        
        # Calculate average attention per layer
        avg_attentions = []
        for layer_att in attentions:
            # layer_att shape: (batch=1, heads, seq_len, seq_len)
            avg_att = layer_att[0].mean(axis=0)  # Average over heads
            avg_attentions.append(avg_att)
        
        # Create subplots for multiple layers
        num_layers = len(avg_attentions)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, (avg_att, ax) in enumerate(zip(avg_attentions, axes)):
            valid_length = len([t for t in tokens if t != '[PAD]'])
            avg_att = avg_att[:valid_length, :valid_length]
            valid_tokens = tokens[:valid_length]
            
            sns.heatmap(
                avg_att,
                xticklabels=valid_tokens if idx == num_layers - 1 else [],
                yticklabels=valid_tokens,
                cmap='viridis',
                cbar=True,
                ax=ax,
                square=True,
                cbar_kws={'label': 'Attention'}
            )
            
            ax.set_title(f'Layer {idx}')
            if idx == num_layers - 1:
                ax.set_xlabel('Key Tokens')
                plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
            ax.set_ylabel('Query Tokens')
            plt.setp(ax.get_yticklabels(), fontsize=6)
        
        # Hide extra subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')
        
        if text:
            fig.suptitle(f'Average Attention per Layer\nText: {text[:100]}...', 
                        fontsize=12, y=0.995)
        else:
            fig.suptitle('Average Attention per Layer', fontsize=12, y=0.995)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{sample_name}_avg_attention.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_attention_flow(self, attention_data, sample_name):
                attentions = attention_data['attentions']
        tokens = attention_data['tokens']
        
        # Extract [CLS] token attention across layers
        cls_attentions = []
        for layer_att in attentions:
            # Average over heads, take [CLS] token (position 0)
            cls_att = layer_att[0, :, 0, :].mean(axis=0)
            cls_attentions.append(cls_att)
        
        cls_attentions = np.array(cls_attentions)  # (num_layers, seq_len)
        
        # Filter valid tokens
        valid_length = len([t for t in tokens if t != '[PAD]'])
        cls_attentions = cls_attentions[:, :valid_length]
        valid_tokens = tokens[:valid_length]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            cls_attentions,
            xticklabels=valid_tokens,
            yticklabels=[f'Layer {i}' for i in range(len(cls_attentions))],
            cmap='YlOrRd',
            cbar=True,
            cbar_kws={'label': 'Attention from [CLS]'}
        )
        
        plt.xlabel('Tokens')
        plt.ylabel('Layer')
        plt.title('[CLS] Token Attention Flow Through Layers')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.tight_layout()
        
        save_path = self.output_dir / f'{sample_name}_cls_flow.png'
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def compute_attention_entropy(self, attention_weights):
                # Add small epsilon to avoid log(0)
        eps = 1e-10
        attention_weights = np.clip(attention_weights, eps, 1.0)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(attention_weights * np.log(attention_weights), axis=-1)
        
        return entropy
    
    def analyze_attention_statistics(self, attention_data):
                attentions = attention_data['attentions']
        tokens = attention_data['tokens']
        
        stats = {
            'num_layers': len(attentions),
            'num_heads': attentions[0].shape[1],
            'sequence_length': attentions[0].shape[-1],
            'layer_stats': []
        }
        
        for layer_idx, layer_att in enumerate(attentions):
            # layer_att: (1, num_heads, seq_len, seq_len)
            layer_att = layer_att[0]  # Remove batch dimension
            
            layer_stat = {
                'layer': layer_idx,
                'mean_attention': float(layer_att.mean()),
                'std_attention': float(layer_att.std()),
                'max_attention': float(layer_att.max()),
                'min_attention': float(layer_att.min()),
                'entropy': []
            }
            
            # Compute entropy for each head
            for head_idx in range(layer_att.shape[0]):
                head_att = layer_att[head_idx]
                entropy = self.compute_attention_entropy(head_att)
                layer_stat['entropy'].append(float(entropy.mean()))
            
            layer_stat['mean_entropy'] = float(np.mean(layer_stat['entropy']))
            stats['layer_stats'].append(layer_stat)
        
        return stats
