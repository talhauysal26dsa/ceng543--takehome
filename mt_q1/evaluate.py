

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from torch.utils.data import DataLoader

from models import BiLSTMClassifier, BiGRUClassifier, BERTClassifier
from utils import IMDbDataLoader
from utils.preprocessing import TextPreprocessor, create_vocabulary
from utils.embedding_loader import EmbeddingLoader
from train import TokenizedDataset


class SentimentEvaluator:
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: torch.device, is_bert: bool = False):
       
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.is_bert = is_bert
        
        self.model.eval()
    
    def evaluate(self) -> Dict:
       
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        # Test loop
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                start_time = time.time()
                
                if self.is_bert:
                    # BERT için
                    texts, labels = batch
                    
                    encoding = self.model.tokenize_texts(texts, max_length=256)
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    # LSTM/GRU için
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    outputs = self.model(input_ids)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                # Collect
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, 
            all_predictions, 
            all_probabilities, 
            inference_times
        )
        
        # Print results
        self._print_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self, 
                          labels: np.ndarray, 
                          predictions: np.ndarray,
                          probabilities: np.ndarray,
                          inference_times: List[float]) -> Dict:
       
        accuracy = accuracy_score(labels, predictions)
        
       
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        
        macro_f1 = f1.mean()
        
        
        cm = confusion_matrix(labels, predictions)
        
        
        avg_inference_time = np.mean(inference_times)
        total_samples = len(labels)
        samples_per_second = total_samples / sum(inference_times)
        
        report = classification_report(
            labels, predictions, 
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'precision_negative': float(precision[0]),
            'precision_positive': float(precision[1]),
            'recall_negative': float(recall[0]),
            'recall_positive': float(recall[1]),
            'f1_negative': float(f1[0]),
            'f1_positive': float(f1[1]),
            'confusion_matrix': cm.tolist(),
            'avg_inference_time': float(avg_inference_time),
            'samples_per_second': float(samples_per_second),
            'total_samples': int(total_samples),
            'classification_report': report
        }
        
        return metrics
    
    def _print_results(self, metrics: Dict):

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        
        print(f"\n Overall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        
        print(f"\n Per-Class Metrics:")
        print(f"\n  Negative (0):")
        print(f"    Precision: {metrics['precision_negative']:.4f}")
        print(f"    Recall:    {metrics['recall_negative']:.4f}")
        print(f"    F1 Score:  {metrics['f1_negative']:.4f}")
        
        print(f"\n  Positive (1):")
        print(f"    Precision: {metrics['precision_positive']:.4f}")
        print(f"    Recall:    {metrics['recall_positive']:.4f}")
        print(f"    F1 Score:  {metrics['f1_positive']:.4f}")
        
        print(f"\n Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                 Predicted")
        print(f"                 Neg    Pos")
        print(f"  Actual  Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"          Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]")
        
        print(f"\n Performance:")
        print(f"  Avg inference time: {metrics['avg_inference_time']*1000:.2f} ms")
        print(f"  Throughput:         {metrics['samples_per_second']:.1f} samples/sec")
        print(f"  Total samples:      {metrics['total_samples']:,}")
        
        print(f"\n{'='*60}")


def compare_models(experiment_dir: str):
   
    experiment_path = Path(experiment_dir)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    
    history_files = list(experiment_path.glob("*/*_history.json"))
    
    if not history_files:
        print(" Henüz eğitilmiş model yok!")
        return
    
    results = []
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
            results.append(history)
    
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n{'Model':<20} {'Val Acc':<10} {'Epochs':<8} {'Avg Time':<12} {'Convergence'}")
    print("-" * 70)
    
    for result in results:
        model_name = result['model_name']
        val_acc = result['best_val_acc']
        epochs = result['total_epochs']
        avg_time = result['avg_epoch_time']
        
        convergence = "N/A"
        for i, acc in enumerate(result['val_accs'], 1):
            if acc >= 0.85:
                convergence = f"{i} epochs"
                break
        
        print(f"{model_name:<20} {val_acc:.4f}    {epochs:<8} {avg_time:>8.2f}s    {convergence}")
    
    print("-" * 70)
    
    best_model = results[0]
    print(f"\n Best Model: {best_model['model_name']}")
    print(f"   Validation Accuracy: {best_model['best_val_acc']:.4f}")
    
    print(f"\n{'='*70}")


def load_model_and_data(model_name: str, checkpoint_path: str, data_dir: str, batch_size: int, device: torch.device):
    
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f" Checkpoint bulunamadı: {checkpoint_path}")
        return None, None, False
    
    # Model tipini belirle
    model_type, embedding_type = model_name.split('_')
    is_bert = (embedding_type == 'bert')
    
    print(f"\n Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f" Checkpoint info:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Train Acc: {checkpoint.get('train_acc', 0):.4f}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 0):.4f}")
    
    # Dataset yükle
    print(f"\n Loading test dataset...")
    data_loader = IMDbDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        max_length=256,
        num_workers=0
    )
    
    test_dataset = data_loader.test_dataset
    
 
    if is_bert:
        print(f"\n Creating BERT model...")
        model = BERTClassifier(
            model_name='distilbert-base-uncased',
            num_classes=2,
            dropout=0.3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        
        model = model.to(device)
        
        test_loader = data_loader.get_test_loader()
        
    else:
       
        print(f"\n Loading GloVe and vocabulary...")
        
        vocab_file = checkpoint_file.parent / 'vocabulary.json'
        if not vocab_file.exists():
            print(f" Vocabulary dosyası bulunamadı: {vocab_file}")
            print("   Model tekrar eğitilmeli (vocabulary kayıt edilmeli).")
            return None, None, False
        
        print(f" Loading vocabulary: {vocab_file}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        word_to_idx = vocab_data['word_to_idx']
        idx_to_word = vocab_data['idx_to_word']
        vocab_size = vocab_data['vocab_size']
        print(f" Vocabulary size: {vocab_size:,}")
        
        # GloVe embeddings yükle
        embedding_path = Path('embeddings/glove/glove.6B.300d.txt')
        if not embedding_path.exists():
            print(f" GloVe embeddings bulunamadı: {embedding_path}")
            return None, None, False
        
        embedding_loader = EmbeddingLoader(str(embedding_path), embedding_dim=300)
        
        embedding_matrix = embedding_loader.create_embedding_matrix(
            word_to_idx,
            trainable_oov=True
        )
        
        # Model oluştur
        print(f"\n Creating {model_type.upper()} model...")
        if model_type == 'lstm':
            from models import BiLSTMClassifier
            model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=256,
                num_layers=2,
                num_classes=2,
                dropout=0.5,
                embedding_matrix=embedding_matrix,
                freeze_embeddings=False
            )
        else:  # gru
            from models import BiGRUClassifier
            model = BiGRUClassifier(
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=256,
                num_layers=2,
                num_classes=2,
                dropout=0.5,
                embedding_matrix=embedding_matrix,
                freeze_embeddings=False
            )
        
        # Checkpoint yükle
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Model'i device'a taşı (checkpoint load'dan sonra)
        model = model.to(device)
        
        # Tokenized dataset oluştur
        from train import TokenizedDataset
        preprocessor = TextPreprocessor(max_length=256)
        
        test_tokenized = TokenizedDataset(
            test_dataset.texts,
            test_dataset.labels,
            word_to_idx,
            preprocessor,
            max_length=256
        )
        
        test_loader = DataLoader(
            test_tokenized,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Modeli device'a taşı
    model = model.to(device)
    model.eval()
    
    return model, test_loader, is_bert


def main():
    """Ana evaluation fonksiyonu."""
    parser = argparse.ArgumentParser(description='Evaluate sentiment classification models')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Model ismi (örn: lstm_glove)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint dosyası')
    parser.add_argument('--data_dir', type=str, default='data/raw/aclImdb',
                       help='IMDb dataset klasörü')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Sonuçları kaydetmek için dosya')
    parser.add_argument('--compare', action='store_true',
                       help='Tüm modelleri karşılaştır')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiment klasörü')
    
    args = parser.parse_args()
    
    # Karşılaştırma modu
    if args.compare:
        compare_models(args.experiment_dir)
        return
    
    # Model evaluation için model ve checkpoint gerekli
    if not args.model or not args.checkpoint:
        print(" --model ve --checkpoint parametreleri gerekli!")
        print("\nÖrnek kullanım:")
        print("  python evaluate.py --model lstm_bert --checkpoint experiments/lstm_bert/lstm_bert_best.pth")
        print("\nVeya tüm modelleri karşılaştır:")
        print("  python evaluate.py --compare")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    # Model ve data yükle
    model, test_loader, is_bert = load_model_and_data(
        args.model,
        args.checkpoint,
        args.data_dir,
        args.batch_size,
        device
    )
    
    if model is None:
        return
    
    # Evaluate
    evaluator = SentimentEvaluator(model, test_loader, device, is_bert)
    metrics = evaluator.evaluate()
    
    # Sonuçları kaydet
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
