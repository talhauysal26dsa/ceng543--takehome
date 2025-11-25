

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BahdanauAttention(nn.Module):
   
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        
        super(BahdanauAttention, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim
        
        # W1: Encoder hidden state'i transform et
        self.W1 = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        
        # W2: Decoder hidden state'i transform et
        self.W2 = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        
        # v: Final linear transformation (skor üret)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # 1. Encoder outputs'u transform et
    
        encoder_transformed = self.W1(encoder_outputs)
        
        # 2. Decoder hidden'ı transform et
        
        decoder_transformed = self.W2(decoder_hidden).unsqueeze(1)
        
        # 3. Addition + tanh (non-linearity)
       
        combined = torch.tanh(encoder_transformed + decoder_transformed)
        
        # 4. Linear transformation → skorlar
        
        scores = self.v(combined)
        
        # 5. Squeeze (son boyutu kaldır)
        
        scores = scores.squeeze(-1)
        
      
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 7. Softmax → attention weights (toplamları 1)
       
        attention_weights = F.softmax(scores, dim=-1)
        
        # 8. Weighted sum (context vector)
    
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, src_len]
            encoder_outputs                   # [batch, src_len, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]
        
        return context, attention_weights


class LuongAttention(nn.Module):
   
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, score_function="general"):
        
        super(LuongAttention, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.score_function = score_function
        
        if score_function == "general":
            # W: Decoder 
            self.W = nn.Linear(decoder_hidden_dim, encoder_hidden_dim, bias=False)
        
        elif score_function == "concat":
            # Bahdanau benzeri 
            self.W = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, 
                              decoder_hidden_dim, bias=False)
            self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Score hesaplama (3 farklı yöntem)
        if self.score_function == "dot":
            scores = torch.bmm(
                encoder_outputs,           
                decoder_hidden.unsqueeze(-1)  
            ).squeeze(-1)  
        
        elif self.score_function == "general":
            # s^T * W * h
            # 1. decoder_hidden → W → [batch, encoder_hidden_dim]
            decoder_transformed = self.W(decoder_hidden)
            
            # 2. Dot product
            scores = torch.bmm(
                encoder_outputs,                     
                decoder_transformed.unsqueeze(-1)    
            ).squeeze(-1)  
        elif self.score_function == "concat":
            # v^T * tanh(W * [s; h])
            
            decoder_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
           
            
            # 2. Concat
            combined = torch.cat([encoder_outputs, decoder_expanded], dim=-1)
            
            
            # 3. Linear + tanh + linear
            scores = self.v(torch.tanh(self.W(combined))).squeeze(-1)
            # [batch, src_len]
        
        # Masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  
            encoder_outputs                   
        ).squeeze(1)  
        
        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
       
        super(ScaledDotProductAttention, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        
        self.query_projection = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        
        
        self.key_projection = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.value_projection = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        
        # Scaling factor
        self.scale = math.sqrt(encoder_hidden_dim)
    
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
       
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # 1. Linear projections
       
        Q = self.query_projection(decoder_hidden)  
        
        # Key: 
        K = self.key_projection(encoder_outputs)   # [batch, src_len, enc_dim]
        
        # Value: 
        V = self.value_projection(encoder_outputs)  # [batch, src_len, enc_dim]
        
        # 2. Scaled dot-product
        scores = torch.bmm(
            Q.unsqueeze(1),          
            K.transpose(1, 2)        
        ) / self.scale  
        
        
        scores = scores.squeeze(1)
        
        # 3. Masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax
        attention_weights = F.softmax(scores, dim=-1)  
        
        # 5. Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  
            V                                 
        ).squeeze(1)  
        
        return context, attention_weights


def test_attention():
    
    
    print("=" * 70)
    print(" Attention Mechanisms Test")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 2
    src_len = 5
    encoder_hidden_dim = 1024  # Bidirectional 
    decoder_hidden_dim = 512
    attention_dim = 512
    
    # Dummy input
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_dim)
    decoder_hidden = torch.randn(batch_size, decoder_hidden_dim)
    
    print(f"\nInput shapes:")
    print(f"   encoder_outputs: {encoder_outputs.shape}")
    print(f"   decoder_hidden: {decoder_hidden.shape}")
    
    
    attentions = {
        "Bahdanau": BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, attention_dim),
        "Luong (general)": LuongAttention(encoder_hidden_dim, decoder_hidden_dim, "general"),
        "Scaled Dot-Product": ScaledDotProductAttention(encoder_hidden_dim, decoder_hidden_dim)
    }
    
    for name, attention in attentions.items():
        print(f"\n" + "=" * 70)
        print(f" Testing: {name}")
        print(f"=" * 70)
        
        # Parametre sayısı
        total_params = sum(p.numel() for p in attention.parameters())
        print(f"   Parameters: {total_params:,}")
        
        # Forward pass
        with torch.no_grad():
            context, weights = attention(encoder_outputs, decoder_hidden)
        
        print(f"\n   Output shapes:")
        print(f"      context: {context.shape}")
        print(f"      attention_weights: {weights.shape}")
        
        # Attention weights analizi
        print(f"\n   Attention weights (batch 0):")
        print(f"      {weights[0].tolist()}")
        print(f"      Sum: {weights[0].sum():.4f} (should be 1.0)")
        print(f"      Max: {weights[0].max():.4f}")
        print(f"      Min: {weights[0].min():.4f}")
        
        # Entropy (sharpness metric)
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
        print(f"      Entropy: {entropy:.4f} (lower = sharper)")
    
    print("\n" + "=" * 70)
    print(" Attention mechanisms test tamamlandı!")
    print("=" * 70)
    
    print("\n Karşılaştırma:")
    print("   Bahdanau:")
    print("      + En güçlü, karmaşık alignment'lar")
    print("      - En fazla parametre, en yavaş")
    print("\n   Luong:")
    print("      + Basit, hızlı")
    print("      - Biraz daha az ekspresif")
    print("\n   Scaled Dot-Product:")
    print("      + Transformer'da kullanılır, çok stabil")
    print("      + Paralelleştirilebilir")
    print("      - Scaling kritik")


if __name__ == "__main__":
    test_attention()

