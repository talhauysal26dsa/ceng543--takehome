import yaml
import sys
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "mt_q1"))

from src.model_loader import ModelLoader
from src.attention_viz import AttentionVisualizer
from src.integrated_gradients import IntegratedGradientsAnalyzer
from src.lime_analyzer import LIMEAnalyzer
from src.error_analysis import ErrorAnalyzer
from src.uncertainty import UncertaintyQuantifier

# Import data loader from mt_q1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mt_q1'))
from utils.data_loader import IMDbDataLoader

def load_config(config_path='config.yaml'):
        with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
        torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_test_data(config):
        print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    # Path should point to the aclImdb directory (use absolute path)
    workspace_root = Path(__file__).parent.parent
    data_path = workspace_root / 'mt_q1' / 'data' / 'raw' / 'aclImdb' / 'test'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Test data directory not found: {data_path}")
    
    # Load test data directly from files
    texts = []
    labels = []
    sample_size = config['data']['sample_size']
    
    # Load positive reviews
    pos_dir = data_path / 'pos'
    if pos_dir.exists():
        pos_files = sorted(pos_dir.glob('*.txt'))[:sample_size//2 if sample_size else None]
        for file_path in pos_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                labels.append(1)
    
    # Load negative reviews
    neg_dir = data_path / 'neg'
    if neg_dir.exists():
        neg_files = sorted(neg_dir.glob('*.txt'))[:sample_size//2 if sample_size else None]
        for file_path in neg_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                labels.append(0)
    
    print(f"✓ Loaded {len(texts)} test samples ({labels.count(1)} positive, {labels.count(0)} negative)")
    
    return texts, labels

def run_attention_analysis(model_loader, attention_viz, test_texts, config):
        print("\n" + "="*60)
    print("ATTENTION VISUALIZATION ANALYSIS")
    print("="*60)
    
    num_samples = min(config['interpretability']['attention']['num_samples'], len(test_texts))
    sample_indices = np.random.choice(len(test_texts), num_samples, replace=False)
    
    attention_results = []
    
    for idx in tqdm(sample_indices, desc="Analyzing attention patterns"):
        text = test_texts[idx]
        
        # Get attention weights
        attention_data = model_loader.get_attention_weights(text)
        
        # Compute statistics
        stats = attention_viz.analyze_attention_statistics(attention_data)
        
        # Visualize for first 5 samples
        if idx < 5:
            # Average attention across layers
            attention_viz.visualize_average_attention(
                attention_data, 
                f'sample_{idx}',
                text[:100]
            )
            
            # Attention flow
            attention_viz.visualize_attention_flow(attention_data, f'sample_{idx}')
        
        attention_results.append({
            'sample_idx': int(idx),
            'text': text[:200] + '...',
            'statistics': stats
        })
    
    # Save results
    results_path = Path(config['output']['results_dir']) / 'attention_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(attention_results, f, indent=2)
    
    print(f"✓ Attention analysis complete. Results saved to {results_path}")
    
    return attention_results

def run_integrated_gradients_analysis(model, tokenizer, test_texts, config):
        print("\n" + "="*60)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("="*60)
    
    ig_analyzer = IntegratedGradientsAnalyzer(model, tokenizer, config)
    
    num_samples = min(config['interpretability']['integrated_gradients']['num_samples'], len(test_texts))
    sample_indices = np.random.choice(len(test_texts), num_samples, replace=False)
    
    ig_results = []
    
    for idx in tqdm(sample_indices, desc="Computing attributions"):
        text = test_texts[idx]
        
        # Compute attributions
        attr_data = ig_analyzer.compute_attributions(
            text,
            n_steps=config['interpretability']['integrated_gradients']['n_steps']
        )
        
        # Visualize for first 5 samples
        if idx < 5:
            ig_analyzer.visualize_attributions(attr_data, f'sample_{idx}')
            ig_analyzer.visualize_top_attributions(attr_data, f'sample_{idx}', top_k=10)
        
        # Store results (without full attributions to save space)
        ig_results.append({
            'sample_idx': int(idx),
            'text': text[:200] + '...',
            'predicted_class': int(attr_data['predicted_class']),
            'confidence': float(attr_data['confidence']),
            'convergence_delta': float(attr_data['convergence_delta'])
        })
    
    # Save results
    results_path = Path(config['output']['results_dir']) / 'ig_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(ig_results, f, indent=2)
    
    print(f"✓ Integrated Gradients analysis complete. Results saved to {results_path}")
    
    return ig_results

def run_lime_analysis(model, tokenizer, test_texts, config):
        print("\n" + "="*60)
    print("LIME ANALYSIS")
    print("="*60)
    
    lime_analyzer = LIMEAnalyzer(model, tokenizer, config)
    
    num_samples = min(config['interpretability']['lime']['num_samples'], len(test_texts))
    sample_indices = np.random.choice(len(test_texts), num_samples, replace=False)
    
    lime_results = []
    
    for idx in tqdm(sample_indices, desc="Generating LIME explanations"):
        text = test_texts[idx]
        
        # Generate explanation
        exp_data = lime_analyzer.explain_instance(
            text,
            num_features=config['interpretability']['lime']['num_features'],
            num_samples=config['interpretability']['lime']['num_perturbations']
        )
        
        # Visualize for first 5 samples
        if idx < 5:
            lime_analyzer.visualize_explanation(exp_data, f'sample_{idx}')
            lime_analyzer.visualize_probability_distribution(exp_data, f'sample_{idx}')
        
        # Store results
        lime_results.append({
            'sample_idx': int(idx),
            'text': text[:200] + '...',
            'predicted_class': int(exp_data['predicted_class']),
            'confidence': float(exp_data['confidence']),
            'feature_weights': exp_data['feature_weights']
        })
    
    # Save results
    results_path = Path(config['output']['results_dir']) / 'lime_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(lime_results, f, indent=2)
    
    print(f"✓ LIME analysis complete. Results saved to {results_path}")
    
    return lime_results

def run_error_analysis(model, tokenizer, test_texts, test_labels, config):
        print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    error_analyzer = ErrorAnalyzer(model, tokenizer, config)
    
    # Get predictions for all test samples
    print("Collecting predictions...")
    predictions = []
    for text in tqdm(test_texts, desc="Predicting"):
        pred_data = error_analyzer.predict_with_confidence(text)
        predictions.append(pred_data['prediction'])
    
    predictions = np.array(predictions)
    test_labels_array = np.array(test_labels)
    
    # Identify failure cases
    print("\nIdentifying failure cases...")
    failure_cases = error_analyzer.identify_failure_cases(
        test_texts,
        predictions,
        test_labels_array
    )
    
    # Analyze patterns
    print("Analyzing failure patterns...")
    analysis = error_analyzer.analyze_failure_patterns(failure_cases)
    
    # Select representative cases
    print("Selecting representative failure cases...")
    representative_cases = error_analyzer.select_representative_failures(
        failure_cases,
        num_per_category=5
    )
    
    # Visualizations
    print("Creating visualizations...")
    error_analyzer.visualize_failure_distribution(failure_cases)
    error_analyzer.visualize_confidence_distribution(failure_cases)
    
    # Generate report
    report_path = error_analyzer.generate_error_report(
        failure_cases,
        analysis,
        representative_cases
    )
    
    print(f"✓ Error analysis complete. Report saved to {report_path}")
    
    return failure_cases, analysis, representative_cases

def run_uncertainty_quantification(model, tokenizer, test_texts, test_labels, config):
        print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*60)
    
    uncertainty_quantifier = UncertaintyQuantifier(model, tokenizer, config)
    
    # Compute predictive entropy
    print("Computing predictive entropy...")
    entropy_data = uncertainty_quantifier.compute_predictive_entropy(
        test_texts,
        batch_size=config['data']['batch_size']
    )
    
    # Visualize entropy
    uncertainty_quantifier.visualize_entropy_distribution(entropy_data)
    
    # Compute calibration
    print("\nComputing calibration metrics...")
    
    # Get all predictions and probabilities
    all_probs = []
    all_preds = []
    
    for text in tqdm(test_texts, desc="Collecting predictions"):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(model.device if hasattr(model, 'device') 
                                             else next(model.parameters()).device)
        attention_mask = encoding['attention_mask'].to(model.device if hasattr(model, 'device')
                                                       else next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            
            all_probs.append(probs)
            all_preds.append(pred)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    test_labels_array = np.array(test_labels)
    
    calibration_data = uncertainty_quantifier.compute_calibration(
        all_probs,
        all_preds,
        test_labels_array,
        n_bins=config['uncertainty']['calibration']['n_bins']
    )
    
    # Visualize calibration
    uncertainty_quantifier.visualize_calibration(calibration_data)
    
    # MC Dropout for selected samples
    if config['uncertainty']['mc_dropout']['enabled']:
        print("\nPerforming Monte Carlo Dropout analysis...")
        mc_samples = test_texts[:5]  # Analyze first 5 samples
        
        for idx, text in enumerate(mc_samples):
            mc_data = uncertainty_quantifier.mc_dropout_predict(
                text,
                num_samples=config['uncertainty']['mc_dropout']['num_samples']
            )
            uncertainty_quantifier.visualize_mc_dropout_uncertainty(
                mc_data,
                Path(config['output']['results_dir']) / 'uncertainty' / f'mc_dropout_sample_{idx}.png'
            )
    
    # Generate report
    report_path = uncertainty_quantifier.generate_uncertainty_report(
        entropy_data,
        calibration_data
    )
    
    print(f"✓ Uncertainty quantification complete. Report saved to {report_path}")
    
    return entropy_data, calibration_data

def main():
        print("\n" + "="*60)
    print("MT Q5: MODEL INTERPRETABILITY AND ERROR ANALYSIS")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create output directories
    for dir_path in [config['output']['experiments_dir'],
                     config['output']['visualizations_dir'],
                     config['output']['results_dir'],
                     config['output']['reports_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading best-performing model (GRU+BERT from mt_q1)...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model()
    
    # Load test data
    test_texts, test_labels = load_test_data(config)
    
    # Initialize analyzers
    attention_viz = AttentionVisualizer(config)
    
    # Run analyses
    attention_results = None
    if config['interpretability']['attention']['enabled']:
        results_path = Path(config['output']['results_dir']) / 'attention_analysis_results.json'
        if results_path.exists():
            print("\n" + "="*60)
            print("ATTENTION VISUALIZATION ANALYSIS")
            print("="*60)
            print(f"✓ Skipping - results already exist at {results_path}")
            with open(results_path, 'r') as f:
                attention_results = json.load(f)
        else:
            attention_results = run_attention_analysis(
                model_loader, attention_viz, test_texts, config
            )
    
    ig_results = None
    if config['interpretability']['integrated_gradients']['enabled']:
        results_path = Path(config['output']['results_dir']) / 'ig_analysis_results.json'
        if results_path.exists():
            print("\n" + "="*60)
            print("INTEGRATED GRADIENTS ANALYSIS")
            print("="*60)
            print(f"✓ Skipping - results already exist at {results_path}")
            with open(results_path, 'r') as f:
                ig_results = json.load(f)
        else:
            ig_results = run_integrated_gradients_analysis(
                model, tokenizer, test_texts, config
            )
    
    lime_results = None
    if config['interpretability']['lime']['enabled']:
        results_path = Path(config['output']['results_dir']) / 'lime_analysis_results.json'
        if results_path.exists():
            print("\n" + "="*60)
            print("LIME ANALYSIS")
            print("="*60)
            print(f"✓ Skipping - results already exist at {results_path}")
            with open(results_path, 'r') as f:
                lime_results = json.load(f)
        else:
            lime_results = run_lime_analysis(
                model, tokenizer, test_texts, config
            )
    
    # Error analysis
    failure_cases, error_analysis, representative_cases = run_error_analysis(
        model, tokenizer, test_texts, test_labels, config
    )
    
    # Uncertainty quantification
    entropy_data, calibration_data = run_uncertainty_quantification(
        model, tokenizer, test_texts, test_labels, config
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Visualizations: {config['output']['visualizations_dir']}")
    print(f"  - Results: {config['output']['results_dir']}")
    print(f"  - Reports: {config['output']['reports_dir']}")
    print("\nRun 'python generate_report.py' to create the final comprehensive report.")

if __name__ == '__main__':
    main()
