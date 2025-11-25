import sys
from pathlib import Path
import yaml
import torch
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "mt_q1"))

from src.model_loader import ModelLoader
from src.attention_viz import AttentionVisualizer
from src.integrated_gradients import IntegratedGradientsAnalyzer
from src.lime_analyzer import LIMEAnalyzer

def quick_example():
        print("="*60)
    print("MT Q5: Quick Example - Single Text Analysis")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Example text
    example_text = (
        "This movie was absolutely amazing! The acting was superb, "
        "the cinematography was breathtaking, and the plot kept me "
        "engaged from start to finish. However, the ending felt a "
        "bit rushed and could have been better developed. Overall, "
        "I highly recommend it!"
    )
    
    print(f"\nAnalyzing text:\n\"{example_text}\"\n")
    
    # Load model
    print("Loading model...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    
    # Get prediction
    print("\n" + "-"*60)
    print("PREDICTION")
    print("-"*60)
    result = model_loader.predict(example_text)
    sentiment = "Positive" if result['prediction'] == 1 else "Negative"
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Negative={result['probabilities'][0]:.4f}, "
          f"Positive={result['probabilities'][1]:.4f}")
    
    # Attention visualization
    print("\n" + "-"*60)
    print("ATTENTION ANALYSIS")
    print("-"*60)
    attention_viz = AttentionVisualizer(config)
    attention_data = model_loader.get_attention_weights(example_text)
    
    print(f"Extracted attention from {len(attention_data['attentions'])} layers")
    print(f"Number of attention heads: {attention_data['attentions'][0].shape[1]}")
    
    # Visualize
    attention_viz.visualize_average_attention(attention_data, 'quick_example', example_text)
    attention_viz.visualize_attention_flow(attention_data, 'quick_example')
    
    stats = attention_viz.analyze_attention_statistics(attention_data)
    num_layers = len(stats['layer_stats'])
    print(f"Mean entropy (Layer 0): {stats['layer_stats'][0]['mean_entropy']:.4f}")
    print(f"Mean entropy (Layer {num_layers-1}): {stats['layer_stats'][num_layers-1]['mean_entropy']:.4f}")
    print("✓ Attention visualizations saved")
    
    # Integrated Gradients
    print("\n" + "-"*60)
    print("INTEGRATED GRADIENTS")
    print("-"*60)
    ig_analyzer = IntegratedGradientsAnalyzer(model, tokenizer, config)
    
    print("Computing attributions (50 integration steps)...")
    attr_data = ig_analyzer.compute_attributions(example_text, n_steps=50)
    
    print(f"Convergence delta: {attr_data['convergence_delta']:.6f}")
    print("\nTop 5 positive contributions:")
    
    # Get top attributions
    tokens = attr_data['tokens']
    attributions = attr_data['attributions']
    
    # Filter special tokens
    valid_indices = [i for i, t in enumerate(tokens) 
                    if t not in ['[CLS]', '[SEP]', '[PAD]']]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_attrs = [attributions[i] for i in valid_indices]
    
    # Sort by attribution
    sorted_indices = sorted(range(len(valid_attrs)), 
                          key=lambda i: valid_attrs[i], reverse=True)
    
    for i in sorted_indices[:5]:
        print(f"  {valid_tokens[i]}: {valid_attrs[i]:.4f}")
    
    print("\nTop 5 negative contributions:")
    for i in sorted_indices[-5:]:
        print(f"  {valid_tokens[i]}: {valid_attrs[i]:.4f}")
    
    # Visualize
    ig_analyzer.visualize_attributions(attr_data, 'quick_example')
    ig_analyzer.visualize_top_attributions(attr_data, 'quick_example', top_k=10)
    print("✓ IG visualizations saved")
    
    # LIME
    print("\n" + "-"*60)
    print("LIME ANALYSIS")
    print("-"*60)
    lime_analyzer = LIMEAnalyzer(model, tokenizer, config)
    
    print("Generating LIME explanation (1000 perturbations)...")
    exp_data = lime_analyzer.explain_instance(
        example_text,
        num_features=10,
        num_samples=1000
    )
    
    print("\nTop feature weights:")
    for feature, weight in exp_data['feature_weights'][:10]:
        direction = "→ Positive" if weight > 0 else "→ Negative"
        print(f"  '{feature}': {weight:.4f} {direction}")
    
    # Visualize
    lime_analyzer.visualize_explanation(exp_data, 'quick_example')
    lime_analyzer.visualize_probability_distribution(exp_data, 'quick_example')
    print("✓ LIME visualizations saved")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nVisualization files saved in:")
    print(f"  - Attention: visualizations/attention/")
    print(f"  - IG: visualizations/integrated_gradients/")
    print(f"  - LIME: visualizations/lime/")
    print("\nCheck these directories for detailed plots!")

if __name__ == '__main__':
    quick_example()
