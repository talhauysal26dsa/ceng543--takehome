import torch
import sys
import os
from pathlib import Path

# Add mt_q1 to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent / "mt_q1"))

from models.bert_classifier import BERTClassifier
from transformers import DistilBertTokenizer

class ModelLoader:
        def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
                print(f"Loading model from: {self.config['model']['checkpoint_path']}")
        
        # Initialize DistilBERT tokenizer and model
        model_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Initialize BERT classifier
        self.model = BERTClassifier(
            model_name=model_name,
            num_classes=2,
            dropout=0.3,
            freeze_bert=False
        )
        
        # Load checkpoint - resolve path relative to workspace root (parent of mt_q5)
        checkpoint_path = Path(self.config['model']['checkpoint_path'])
        if not checkpoint_path.is_absolute():
            # If relative, make it relative to workspace root (parent directory of mt_q5)
            workspace_root = Path(__file__).parent.parent.parent
            checkpoint_path = workspace_root / checkpoint_path
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✓ Model loaded successfully")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        return self.model, self.tokenizer
    
    def prepare_input(self, text, max_length=512):
                encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, text):
                inputs = self.prepare_input(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0]
        }
    
    def get_attention_weights(self, text):
                inputs = self.prepare_input(text)
        
        with torch.no_grad():
            # Get BERT outputs with attention
            bert_outputs = self.model.bert(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True
            )
            
            # Extract attention weights from all layers
            attentions = bert_outputs.attentions  # Tuple of (batch, heads, seq, seq)
            
        return {
            'attentions': [att.cpu().numpy() for att in attentions],
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        }
