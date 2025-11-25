

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from models import BiLSTMClassifier, BiGRUClassifier, BERTClassifier
from evaluate import load_model_and_data

# Stil ayarları
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 10


def extract_hidden_representations(model, test_loader, device, is_bert, max_samples=2000):
   
    model.eval()
    
    all_representations = []
    all_labels = []
    total_samples = 0
    
    print(f"\n Extracting hidden representations (max {max_samples} samples)...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            if total_samples >= max_samples:
                break
            
            if is_bert:
                # BERT için
                texts, labels = batch
                labels = labels.to(device) 
                
                encoding = model.tokenize_texts(texts, max_length=256)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # BERT hidden state al
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                hidden_state = outputs.last_hidden_state
                # [CLS] token'ın representation'ı
                cls_repr = hidden_state[:, 0, :].cpu().numpy()
                
                # Labels'ı CPU'ya geri al
                labels = labels.cpu()
                
            else:
                # LSTM/GRU için
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Embedding + RNN
                embedded = model.embedding(input_ids)
                
                if hasattr(model, 'lstm'):
                    output, (hidden, cell) = model.lstm(embedded)
                else:  # GRU
                    output, hidden = model.gru(embedded)
                
                # Son layer'ın hidden state'i
                if model.bidirectional:
                    # Forward + Backward concat
                    hidden_forward = hidden[-2, :, :]
                    hidden_backward = hidden[-1, :, :]
                    hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
                else:
                    hidden_concat = hidden[-1, :, :]
                
                cls_repr = hidden_concat.cpu().numpy()
                labels = labels.cpu()
            
            all_representations.append(cls_repr)
            all_labels.extend(labels.numpy())
            total_samples += len(labels)
    
    # Concatenate
    representations = np.vstack(all_representations)[:max_samples]
    labels = np.array(all_labels)[:max_samples]
    
    print(f"  Extracted {representations.shape[0]} representations")
    print(f"  Shape: {representations.shape}")
    print(f"  Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")
    
    return representations, labels


def apply_pca(representations, n_components=2):
   
    print(f"\n Applying PCA (n_components={n_components})...")
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(representations)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  PCA completed")
    print(f"  Explained variance: {explained_var}")
    print(f"  Total explained: {np.sum(explained_var):.4f} ({np.sum(explained_var)*100:.2f}%)")
    
    return reduced, pca


def apply_tsne(representations, n_components=2, perplexity=30, random_state=42):
    
    print(f"\n Applying t-SNE (perplexity={perplexity})...")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_iter=1000, verbose=1)
    reduced = tsne.fit_transform(representations)
    
    print(f" t-SNE completed")
    
    return reduced


def plot_2d_projection(reduced, labels, method_name, model_name, output_dir):
    
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Positive ve negative ayrı plot et
    mask_pos = labels == 1
    mask_neg = labels == 0
    
    ax.scatter(reduced[mask_neg, 0], reduced[mask_neg, 1], 
              c='blue', label='Negative', alpha=0.5, s=20, edgecolors='none')
    ax.scatter(reduced[mask_pos, 0], reduced[mask_pos, 1], 
              c='red', label='Positive', alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
    ax.set_title(f'{method_name} Projection - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    output_file = Path(output_dir) / f"{model_name}_{method_name.lower()}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def plot_combined_comparison(pca_reduced, tsne_reduced, labels, model_name, output_dir):
    
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Masks
    mask_pos = labels == 1
    mask_neg = labels == 0
    
    # PCA
    ax = axes[0]
    ax.scatter(pca_reduced[mask_neg, 0], pca_reduced[mask_neg, 1], 
              c='blue', label='Negative', alpha=0.5, s=20, edgecolors='none')
    ax.scatter(pca_reduced[mask_pos, 0], pca_reduced[mask_pos, 1], 
              c='red', label='Positive', alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('PCA Projection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    ax = axes[1]
    ax.scatter(tsne_reduced[mask_neg, 0], tsne_reduced[mask_neg, 1], 
              c='blue', label='Negative', alpha=0.5, s=20, edgecolors='none')
    ax.scatter(tsne_reduced[mask_pos, 0], tsne_reduced[mask_pos, 1], 
              c='red', label='Positive', alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Projection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Latent Space Visualization - {model_name}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Kaydet
    output_file = Path(output_dir) / f"{model_name}_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    
    plt.close()


def analyze_separability(representations, labels):
    
    print("\n  Analyzing class separability...")
    
    # Her sınıfın merkezi
    pos_repr = representations[labels == 1]
    neg_repr = representations[labels == 0]
    
    pos_center = np.mean(pos_repr, axis=0)
    neg_center = np.mean(neg_repr, axis=0)
    
    # Merkezler arası mesafe
    center_distance = np.linalg.norm(pos_center - neg_center)
    
    # Sınıf içi varyans
    pos_var = np.mean(np.var(pos_repr, axis=0))
    neg_var = np.mean(np.var(neg_repr, axis=0))
    avg_var = (pos_var + neg_var) / 2
    
    # Separability score (daha yüksek = daha iyi ayrılabilir)
    separability = center_distance / np.sqrt(avg_var)
    
    print(f"\n  Center distance: {center_distance:.4f}")
    print(f"  Avg within-class variance: {avg_var:.4f}")
    print(f"  Separability score: {separability:.4f}")
    print(f"    (Higher is better - classes are more separated)")
    
    return {
        'center_distance': float(center_distance),
        'avg_variance': float(avg_var),
        'separability_score': float(separability)
    }


def visualize_model(model_name, checkpoint_path, output_dir, max_samples=2000):
    
    print("\n" + "="*70)
    print(f"VISUALIZING: {model_name}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model ve data yükle
    model, test_loader, is_bert = load_model_and_data(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        data_dir='data/raw/aclImdb',
        batch_size=32,
        device=device
    )
    
    if model is None:
        print(f"  {model_name} visualization atlandı")
        return None
    
    # Modeli GPU'ya taşı (emin olmak için)
    model = model.to(device)
    model.eval()
    
    # Hidden representations çıkar
    representations, labels = extract_hidden_representations(
        model, test_loader, device, is_bert, max_samples
    )
    
    # Separability analizi
    separability_metrics = analyze_separability(representations, labels)
    
    # PCA uygula
    pca_reduced, pca_model = apply_pca(representations)
    plot_2d_projection(pca_reduced, labels, 'PCA', model_name, output_dir)
    
    # t-SNE uygula
    tsne_reduced = apply_tsne(representations, perplexity=30)
    plot_2d_projection(tsne_reduced, labels, 't-SNE', model_name, output_dir)
    
    # Combined plot
    plot_combined_comparison(pca_reduced, tsne_reduced, labels, model_name, output_dir)
    
    print(f"\n  {model_name} visualization completed!")
    
    return {
        'separability': separability_metrics,
        'pca_explained_variance': pca_model.explained_variance_ratio_.tolist()
    }


def main():
   
    parser = argparse.ArgumentParser(description='Visualize latent representations')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Model ismi (örn: lstm_bert). Boş ise tüm modeller.')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiment klasörü')
    parser.add_argument('--output_dir', type=str, default='experiments/visualizations',
                       help='Çıktı klasörü')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maksimum sample sayısı')
    
    args = parser.parse_args()
    
    # Output klasörü oluştur
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("LATENT REPRESENTATION VISUALIZATION")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {args.max_samples}")
    
    if args.model:
        # Tek model
        checkpoint_path = Path(args.experiment_dir) / args.model / f"{args.model}_best.pth"
        if not checkpoint_path.exists():
            print(f" Checkpoint bulunamadı: {checkpoint_path}")
            return
        
        visualize_model(args.model, str(checkpoint_path), output_dir, args.max_samples)
    
    else:
        
        experiment_path = Path(args.experiment_dir)
        checkpoints = list(experiment_path.glob("*/*_best.pth"))
        
        if not checkpoints:
            print(" Hiç eğitilmiş model bulunamadı!")
            return
        
        print(f"\n✓ {len(checkpoints)} model bulundu")
        
        all_metrics = {}
        
        for checkpoint_path in checkpoints:
            model_name = checkpoint_path.parent.name
            
            metrics = visualize_model(model_name, str(checkpoint_path), output_dir, args.max_samples)
            
            if metrics:
                all_metrics[model_name] = metrics
        
      
        metrics_file = output_dir / 'visualization_metrics.json'
        import json
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n✓ Visualization metrics saved: {metrics_file}")
    
    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETED!")
    print("="*70)
    print(f"\nGrafikler: {output_dir}/")
    print("\nHer model için 3 grafik oluşturuldu:")
    print("  - *_pca.png: PCA projection")
    print("  - *_tsne.png: t-SNE projection")
    print("  - *_combined.png: PCA ve t-SNE yan yana")


if __name__ == "__main__":
    main()




