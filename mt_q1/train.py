

import argparse
import os
import json
import time
import yaml
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from models import BiLSTMClassifier, BiGRUClassifier, BERTClassifier
from utils import IMDbDataLoader, EmbeddingLoader
from utils.preprocessing import TextPreprocessor, create_vocabulary


def load_config(config_path: str = 'config.yaml') -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        print(f" Config dosyası bulunamadı: {config_path}")
        print("   Default değerler kullanılacak.")
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f" Config yüklendi: {config_path}")
    return config


class TokenizedDataset(Dataset):
    
    def __init__(self, texts, labels, word_to_idx, preprocessor, max_length=256):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.preprocessor = preprocessor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        
        tokens = self.preprocessor.preprocess(text, pad=True)
        token_ids = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0)) 
                    for token in tokens[:self.max_length]]
        
        
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class SentimentTrainer:
   
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 experiment_dir: str,
                 model_name: str):
    
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name
        
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
      
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
       
        self.epoch_times = []
    
    def train_epoch(self) -> Tuple[float, float]:
       
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            if isinstance(self.model, BERTClassifier):
                
                texts, labels = batch
                labels = labels.to(self.device)
                
                encoding = self.model.tokenize_texts(texts, max_length=256)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
            else:
                
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids)
            
            # Loss hesapla
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Progress bar update
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
       
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if isinstance(self.model, BERTClassifier):
                    # BERT için
                    texts, labels = batch
                    labels = labels.to(self.device)
                    
                    encoding = self.model.tokenize_texts(texts, max_length=256)
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    # LSTM/GRU için
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(input_ids)
                
                # Loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, early_stopping_patience: int = 5):
       
        print(f"\n{'='*60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Training
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            epoch_time = time.time() - start_time
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Metrics kaydet
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.epoch_times.append(epoch_time)
            
            # Sonuçları yazdır
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Best model kaydet
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  Best model saved! (Val Acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n Early stopping triggered! (patience={early_stopping_patience})")
                break
        
        # Final summary
        self.print_training_summary()
        self.save_training_history()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Model checkpoint kaydet."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'train_acc': self.train_accs[-1],
            'val_acc': self.val_accs[-1],
        }
        
        if is_best:
            path = self.experiment_dir / f"{self.model_name}_best.pth"
        else:
            path = self.experiment_dir / f"{self.model_name}_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
    
    def save_training_history(self):
        
        history = {
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'total_epochs': len(self.train_losses),
            'avg_epoch_time': float(np.mean(self.epoch_times))
        }
        
        path = self.experiment_dir / f"{self.model_name}_history.json"
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n Training history saved: {path}")
    
    def print_training_summary(self):
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"Best val accuracy: {self.best_val_acc:.4f}")
        if self.train_accs:
            print(f"Final train accuracy: {self.train_accs[-1]:.4f}")
            print(f"Final val accuracy: {self.val_accs[-1]:.4f}")
        if self.epoch_times:
            print(f"Avg epoch time: {np.mean(self.epoch_times):.2f}s")
            print(f"Total training time: {sum(self.epoch_times)/60:.2f} min")
        print(f"{'='*60}\n")


def train_model(model_type, embedding_type, args, device):
    
    
    model_name = f"{model_type}_{embedding_type}"
    experiment_dir = Path(args.experiment_dir) / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# {model_name.upper()}")
    print(f"{'#'*60}")
    
    # Data loader
    print("\nLoading dataset...")
    data_loader = IMDbDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=256,
        max_samples=args.max_samples,
        num_workers=0
    )
    
    train_dataset = data_loader.train_dataset
    test_dataset = data_loader.test_dataset
    
    # BERT için
    if embedding_type == 'bert':
        print("\nCreating BERT model...")
        model = BERTClassifier(
            model_name='distilbert-base-uncased',
            num_classes=2,
            dropout=args.dropout
        )
        
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_test_loader()
        
    else:
        # GloVe için
        print("\nLoading GloVe embeddings...")
        embedding_path = Path(args.embedding_path)
        if not embedding_path.exists():
            # Relative path dene
            embedding_path = Path(__file__).parent / embedding_path
            if not embedding_path.exists():
                print(f"  GloVe embeddings bulunamadı: {args.embedding_path}")
                print("   Lütfen önce indirin: python utils/download_glove.py")
                return
        
        embedding_loader = EmbeddingLoader(str(embedding_path), embedding_dim=300)
        
        # Vocabulary oluştur
        print("\nCreating vocabulary...")
        preprocessor = TextPreprocessor(max_length=256)
        tokenized_texts = [preprocessor.preprocess(text, pad=False) 
                          for text in train_dataset.texts[:10000]]  # İlk 10k için vocabulary
        
        word_to_idx, idx_to_word = create_vocabulary(
            tokenized_texts,
            min_freq=2,
            max_vocab_size=50000
        )
        
        vocab_size = len(word_to_idx)
        print(f" Vocabulary size: {vocab_size:,}")
        
        # Vocabulary'yi kaydet 
        vocab_file = experiment_dir / 'vocabulary.json'
        vocab_data = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': vocab_size
        }
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f" Vocabulary saved: {vocab_file}")
        
        # Embedding matrix oluştur
        print("\n Creating embedding matrix...")
        embedding_matrix = embedding_loader.create_embedding_matrix(
            word_to_idx,
            trainable_oov=True
        )
        
        # Model oluştur
        print("\nCreating model...")
        if model_type == 'lstm':
            model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=2,
                dropout=args.dropout,
                embedding_matrix=embedding_matrix,
                freeze_embeddings=False
            )
        else:  # gru
            model = BiGRUClassifier(
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=2,
                dropout=args.dropout,
                embedding_matrix=embedding_matrix,
                freeze_embeddings=False
            )
        
        # Tokenized dataset oluştur
        print("\n Creating tokenized datasets...")
        train_tokenized = TokenizedDataset(
            train_dataset.texts,
            train_dataset.labels,
            word_to_idx,
            preprocessor,
            max_length=256
        )
        test_tokenized = TokenizedDataset(
            test_dataset.texts,
            test_dataset.labels,
            word_to_idx,
            preprocessor,
            max_length=256
        )
        
        train_loader = DataLoader(
            train_tokenized,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            test_tokenized,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    
    
    if embedding_type == 'bert':
        # BERT fine-tuning için çok daha düşük learning rate
        bert_lr = 2e-5  # Config'ten gelebilir
        optimizer = optim.AdamW(model.parameters(), lr=bert_lr, weight_decay=0.01)
    else:
        # GloVe için normal learning rate
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    # Trainer
    trainer = SentimentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_dir=str(experiment_dir),
        model_name=model_name
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, early_stopping_patience=5)
    
    print(f"\n{model_name} eğitimi tamamlandı!")
    print(f"Sonuçlar: {experiment_dir}/")


def main():
    
    parser = argparse.ArgumentParser(description='Train sentiment classification models')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config dosyası yolu')
    
    # Model seçimi
    parser.add_argument('--model', type=str, default=None,
                       choices=['lstm', 'gru', 'all'],
                       help='Model tipi (config override eder)')
    parser.add_argument('--embedding', type=str, default=None,
                       choices=['glove', 'bert', 'all'],
                       help='Embedding tipi (config override eder)')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default=None,
                       help='IMDb dataset klasörü (config override eder)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maksimum sample sayısı (config override eder)')
    
    # Embedding
    parser.add_argument('--embedding_path', type=str, default=None,
                       help='GloVe embedding dosyası (config override eder)')
    
    # Training parametreleri
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (config override eder)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Epoch sayısı (config override eder)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (config override eder)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (config override eder)')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='RNN layer sayısı (config override eder)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (config override eder)')
    
    # Experiment
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='Experiment klasörü (config override eder)')
    
    args = parser.parse_args()
    
    # Config yükle
    config = load_config(args.config)
    
    # Config'den değerleri al, args override eder
    # Dataset
    data_dir = args.data_dir if args.data_dir is not None else config.get('dataset', {}).get('path', 'data/raw/aclImdb')
    max_samples = args.max_samples if args.max_samples is not None else config.get('dataset', {}).get('max_samples', None)
    
    # Embeddings
    embedding_path = args.embedding_path if args.embedding_path is not None else config.get('glove', {}).get('path', 'embeddings/glove/glove.6B.300d.txt')
    
    # Training
    training_config = config.get('training', {})
    batch_size = args.batch_size if args.batch_size is not None else training_config.get('batch_size', 32)
    epochs = args.epochs if args.epochs is not None else training_config.get('num_epochs', 10)
    lr = args.lr if args.lr is not None else training_config.get('learning_rate', 0.001)
    
    # Model
    models_config = config.get('models', {})
    lstm_config = models_config.get('lstm', {})
    gru_config = models_config.get('gru', {})
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else lstm_config.get('hidden_dim', 256)
    num_layers = args.num_layers if args.num_layers is not None else lstm_config.get('num_layers', 2)
    dropout = args.dropout if args.dropout is not None else lstm_config.get('dropout', 0.5)
    
    # Experiment
    experiment_dir = args.experiment_dir if args.experiment_dir is not None else config.get('experiments', {}).get('output_dir', 'experiments')
    
    # Model ve embedding seçimi
    model_choice = args.model if args.model is not None else 'all'
    embedding_choice = args.embedding if args.embedding is not None else 'all'
    
    # Args objesi oluştur
    class TrainingArgs:
        pass
    
    training_args = TrainingArgs()
    training_args.data_dir = data_dir
    training_args.embedding_path = embedding_path
    training_args.batch_size = batch_size
    training_args.epochs = epochs
    training_args.lr = lr
    training_args.hidden_dim = hidden_dim
    training_args.num_layers = num_layers
    training_args.dropout = dropout
    training_args.max_samples = max_samples
    training_args.experiment_dir = experiment_dir
    
    args = training_args
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Data dir:       {args.data_dir}")
    print(f"Embedding path: {args.embedding_path}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Hidden dim:     {args.hidden_dim}")
    print(f"Num layers:     {args.num_layers}")
    print(f"Dropout:        {args.dropout}")
    print(f"Max samples:    {args.max_samples if args.max_samples else 'All'}")
    print(f"Experiment dir: {args.experiment_dir}")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Hangi modelleri eğiteceğiz?
    models_to_train = []
    
    if model_choice == 'all':
        model_types = ['lstm', 'gru']
    else:
        model_types = [model_choice]
    
    if embedding_choice == 'all':
        embedding_types = ['glove', 'bert']
    else:
        embedding_types = [embedding_choice]
    
    # Kombinasyonlar
    for model_type in model_types:
        for embedding_type in embedding_types:
            models_to_train.append((model_type, embedding_type))
    
    print(f"\nEğitilecek modeller: {len(models_to_train)}")
    for model_type, embedding_type in models_to_train:
        print(f"  - {model_type.upper()} + {embedding_type.upper()}")
    
    # Her model için eğitim
    for model_type, embedding_type in models_to_train:
        train_model(model_type, embedding_type, args, device)
    
    print(f"\n{'='*60}")
    print("TÜM MODELLERİN EĞİTİMİ TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"Sonuçlar: {args.experiment_dir}/")
    print("Değerlendirme için: python evaluate.py")


if __name__ == "__main__":
    main()
