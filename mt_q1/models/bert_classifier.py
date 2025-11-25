"""
DistilBERT-based sentiment classifier
"""

import warnings
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')


class BERTClassifier(nn.Module):
    
    def __init__(self,
                 model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 freeze_bert: bool = False):
       
        super(BERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # DistilBERT upload
        print(f"\n📥 Loading {model_name}...")
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # DistilBERT Freeze
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("  → DistilBERT frozen (sadece classification head eğitilecek)")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification Head
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_size, num_classes)
        
       
        self._print_model_info()
    
    def _print_model_info(self):
        print(f"\n{'='*60}")
        print("DistilBERT CLASSIFIER")
        print(f"{'='*60}")
        print(f"Model:           {self.model_name}")
        print(f"Hidden size:     {self.bert.config.hidden_size}")
        print(f"Num layers:      {self.bert.config.num_hidden_layers}")
        print(f"Attention heads: {self.bert.config.num_attention_heads}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if trainable_params < total_params:
            frozen_params = total_params - trainable_params
            print(f"Frozen parameters:    {frozen_params:,}")
    
    def forward(self, input_ids, attention_mask=None):
     
        # DistilBERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # last_hidden_state: [batch, seq_len, hidden_size]
        hidden_state = outputs.last_hidden_state
        
        # [batch, hidden_size]
        cls_output = hidden_state[:, 0, :]
        
        # Dropout
        cls_output = self.dropout(cls_output)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def tokenize_texts(self, texts, max_length=512):
      
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,  
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def predict_sentiment(self, text):
      
        self.eval()
        
        # Tokenize
        encoding = self.tokenize_texts([text])
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Predict
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            probability = probs[0, label].item()
        
        label_str = "Positive" if label == 1 else "Negative"
        
        return label_str, probability, logits[0].tolist()


if __name__ == "__main__":
    print("Testing DistilBERT Classifier...")
  
    model = BERTClassifier(
        model_name='distilbert-base-uncased',
        num_classes=2,
        dropout=0.3
    )
    
    # Test sentences
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "It was okay, nothing special but not bad either."
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT PREDICTION TEST")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        label, prob, logits = model.predict_sentiment(text)
        print(f"   Prediction: {label} ({prob*100:.1f}% confidence)")
        print(f"   Logits: [{logits[0]:.3f}, {logits[1]:.3f}]")
    
    print("\n" + "="*60)
    
    # Tokenization example
    print("\nTOKENIZATION EXAMPLE:")
    print("="*60)
    
    sample_text = "This movie was great!"
    encoding = model.tokenize_texts([sample_text])
    
    print(f"Original text: {sample_text}")
    print(f"\nTokenized:")
    print(f"  Input IDs shape: {encoding['input_ids'].shape}")
    print(f"  Input IDs: {encoding['input_ids'][0][:20].tolist()}...")
    
    # Token'ları decode et
    tokens = model.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0][:10])
    print(f"  Tokens: {tokens}")
    
    print("\n  [CLS] → sentence beginning")
    print("  [SEP] → sentence ending")
    print("  [PAD] → Padding")
    
    print("="*60)
