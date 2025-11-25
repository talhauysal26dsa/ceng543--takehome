

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
  
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 embedding_matrix: torch.Tensor = None,
                 freeze_embeddings: bool = False,
                 bidirectional: bool = True):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
        # Model bilgisi
        self._print_model_info()
    
    def _print_model_info(self):
        """Model bilgilerini yazdır."""
        print(f"\n{'='*60}")
        print("BiLSTM MODEL")
        print(f"{'='*60}")
        print(f"Embedding dim:    {self.embedding.embedding_dim}")
        print(f"Hidden dim:       {self.hidden_dim}")
        print(f"Num layers:       {self.num_layers}")
        print(f"Bidirectional:    {self.bidirectional}")
        print(f"Vocabulary size:  {self.embedding.num_embeddings:,}")
        
        # Parametre sayısı
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def forward(self, input_ids, lengths=None):
        batch_size = input_ids.size(0)
        
        # 1. Embedding
       
        embedded = self.embedding(input_ids)
        
        # 2. LSTM
      
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # Pack
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True
            )
            
            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            # Lengths verilmemişse normal forward
            output, (hidden, cell) = self.lstm(embedded)
        
        # 3. Hidden state'i al

        
        if self.bidirectional:
            # Son layer'ın forward ve backward hidden state'lerini concat et
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            # Sadece son layer
            hidden_concat = hidden[-1, :, :]
        
        # 4. Dropout
        hidden_concat = self.dropout(hidden_concat)
        
        # 5. Fully Connected
        logits = self.fc(hidden_concat)
        
        return logits
    
    def get_attention_weights(self, input_ids):
        with torch.no_grad():
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            
            # Her position'ın norm'u → önem göstergesi
            attention = torch.norm(output, dim=2)
            # Normalize et
            attention = attention / attention.sum(dim=1, keepdim=True)
            
            return attention


if __name__ == "__main__":
    # Test
    print("Testing BiLSTMClassifier...")
    
    # Model oluştur
    model = BiLSTMClassifier(
        vocab_size=10000,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.5
    )
    
    # Dummy input
    batch_size = 4
    seq_len = 50
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    output = model(input_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Softmax ile probability'ye çevir
    probs = torch.softmax(output, dim=1)
    print(f"\nProbabilities:\n{probs}")
    
    # Tahmin
    predictions = torch.argmax(probs, dim=1)
    print(f"\nPredictions: {predictions}")
    print(f"  (0=Negative, 1=Positive)")

