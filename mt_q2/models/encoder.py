
import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # 1. Embedding Layer
       
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Padding token ID'si (genellikle 0)
        )
        
        # 2. Dropout Layer
        
        self.dropout = nn.Dropout(dropout)
        
        # 3. Bidirectional GRU
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  
            bidirectional=True,
            batch_first=True  
        )
    
    def forward(self, src, src_lengths):
       
        # 1. Embedding
        # [batch_size, src_len] → [batch_size, src_len, embedding_dim]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        
        # 2. Pack padded sequence
        # Padding'i ignore etmek için PyTorch'un özel formatı
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            src_lengths.cpu(),  # Lengths CPU'da olmalı
            batch_first=True,
            enforce_sorted=False  # Uzunluklara göre sıralanmış olması gerekmez
        )
        
        # 3. GRU forward pass
        # packed_embedded → GRU → packed_outputs
        packed_outputs, hidden = self.gru(packed_embedded)
        
        # 4. Unpack sequence
        # Packed format'tan normal tensor'e geri çevir
        # [batch_size, src_len, hidden_dim*2]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )
        
        # outputs: [batch_size, src_len, hidden_dim*2]
       
        
        return outputs, hidden
    
    def init_weights(self):
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def test_encoder():
    
    
    print("=" * 70)
    print(" Encoder Test")
    print("=" * 70)
    
    # Hyperparameters
    vocab_size = 1000
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    
    # Model oluştur
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    encoder.init_weights()
    
    print(f"\n Model Architecture:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Num layers: {num_layers}")
    print(f"   Dropout: {dropout}")
    
    # Toplam parametre sayısı
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test input
    batch_size = 2
    src_len = 5
    
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    src_lengths = torch.tensor([5, 3])  # İlk cümle 5 kelime, ikinci 3 kelime
    
    print(f"\n Input:")
    print(f"   src shape: {src.shape}")
    print(f"   src_lengths: {src_lengths.tolist()}")
    print(f"\n   src[0] (5 kelime): {src[0].tolist()}")
    print(f"   src[1] (3 kelime + 2 padding): {src[1].tolist()}")
    
    # Forward pass
    with torch.no_grad():  # Gradient hesaplama
        encoder.eval()  # Evaluation mode (dropout kapalı)
        outputs, hidden = encoder(src, src_lengths)
    
    print(f"\n Output:")
    print(f"   outputs shape: {outputs.shape}")  # [batch_size, src_len, hidden_dim*2]
    print(f"   hidden shape: {hidden.shape}")    # [num_layers*2, batch_size, hidden_dim]
    
    # Hidden state'leri açıkla
    print(f"\n Hidden State Detayları:")
    print(f"   num_layers = {num_layers}, bidirectional = True")
    print(f"   → hidden.shape[0] = {hidden.shape[0]} (num_layers * 2)")
    print(f"\n   Layer 1 Forward:  hidden[0] → shape {hidden[0].shape}")
    print(f"   Layer 1 Backward: hidden[1] → shape {hidden[1].shape}")
    print(f"   Layer 2 Forward:  hidden[2] → shape {hidden[2].shape}")
    print(f"   Layer 2 Backward: hidden[3] → shape {hidden[3].shape}")
    
    # Outputs'u açıkla
    print(f"\n Outputs Detayları:")
    print(f"   outputs.shape = [batch_size, src_len, hidden_dim*2]")
    print(f"   → outputs.shape = {outputs.shape}")
    print(f"\n   Her kelime için:")
    print(f"      Forward hidden:  outputs[:, :, :{hidden_dim}]")
    print(f"      Backward hidden: outputs[:, :, {hidden_dim}:]")
    
    # Bidirectional olduğunu kontrol et
    print(f"\n Bidirectional Check:")
    forward_part = outputs[:, :, :hidden_dim]
    backward_part = outputs[:, :, hidden_dim:]
    print(f"   Forward part shape: {forward_part.shape}")
    print(f"   Backward part shape: {backward_part.shape}")
    
    # İlk kelime için forward ve backward hidden state'lerin farklı olduğunu göster
    print(f"\n   İlk kelime (index=0) için:")
    print(f"      Forward hidden (ilk 5 değer):  {forward_part[0, 0, :5].tolist()}")
    print(f"      Backward hidden (ilk 5 değer): {backward_part[0, 0, :5].tolist()}")
    print(f"      → Farklı değerler! (Forward: soldan, Backward: sağdan okur)")
    
    print("\n" + "=" * 70)
    print(" Encoder test tamamlandı!")
    print("=" * 70)
    
    return encoder, outputs, hidden


if __name__ == "__main__":
    test_encoder()

