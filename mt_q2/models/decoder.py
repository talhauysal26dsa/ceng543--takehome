

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import BahdanauAttention, LuongAttention, ScaledDotProductAttention


class Decoder(nn.Module):
    
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 encoder_hidden_dim, attention_type="bahdanau",
                 num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 2. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 3. Attention Mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(
                encoder_hidden_dim=encoder_hidden_dim,
                decoder_hidden_dim=hidden_dim,
                attention_dim=hidden_dim
            )
        elif attention_type == "luong":
            self.attention = LuongAttention(
                encoder_hidden_dim=encoder_hidden_dim,
                decoder_hidden_dim=hidden_dim,
                score_function="general"
            )
        elif attention_type == "scaled_dot":
            self.attention = ScaledDotProductAttention(
                encoder_hidden_dim=encoder_hidden_dim,
                decoder_hidden_dim=hidden_dim
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # 4. GRU
        
        self.gru = nn.GRU(
            input_size=embedding_dim + encoder_hidden_dim,  # embedding + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        
        self.hidden_projection = nn.Linear(
            encoder_hidden_dim,
            hidden_dim
        )

        # 5. Output Layer
        # GRU output + context → vocabulary probabilities
        self.fc_out = nn.Linear(
            hidden_dim + encoder_hidden_dim + embedding_dim,
            vocab_size
        )
    
    def forward(self, input, hidden, encoder_outputs, mask=None):
        
        
        batch_size = input.size(0)
        
        # 1. Embedding
        
        embedded = self.embedding(input).unsqueeze(1)
        embedded = self.dropout(embedded)
        
        # 2. Attention
        
        decoder_hidden_last = hidden[-1]
        
        context, attention_weights = self.attention(
            encoder_outputs,
            decoder_hidden_last,
            mask
        )
        
        # 3. Concat: embedding + context
        
        context = context.unsqueeze(1)
        
        
        gru_input = torch.cat([embedded, context], dim=-1)
        
        # 4. GRU
        # gru_input: [batch_size, 1, embedding_dim + encoder_hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]

        gru_output, hidden = self.gru(gru_input, hidden)
        
        # 5. Output layer
        # Concat: gru_output + context + embedded
        # [batch_size, 1, hidden_dim + encoder_hidden_dim + embedding_dim]
        output_input = torch.cat([
            gru_output,          # [batch, 1, hidden_dim]
            context,             # [batch, 1, encoder_hidden_dim]
            embedded             # [batch, 1, embedding_dim]
        ], dim=-1)
        
        # [batch_size, 1, hidden_dim + encoder_hidden_dim + embedding_dim]
        # → [batch_size, 1, vocab_size]
        output = self.fc_out(output_input)
        
        # Squeeze
        # [batch_size, 1, vocab_size] → [batch_size, vocab_size]
        output = output.squeeze(1)
        attention_weights = attention_weights  # Already [batch_size, src_len]
        
        return output, hidden, attention_weights
    
    def init_hidden(self, encoder_hidden):
       
        # Encoder hidden: [num_layers*2, batch_size, hidden_dim]
       
        
        batch_size = encoder_hidden.size(1)
        encoder_hidden = encoder_hidden.view(
            self.num_layers, 2, batch_size, -1
        )
        
        
        decoder_hidden = torch.cat([
            encoder_hidden[:, 0, :, :],  
            encoder_hidden[:, 1, :, :]   
        ], dim=-1)
        
       
        decoder_hidden = self.hidden_projection(decoder_hidden)
        
        return decoder_hidden
    
    def init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def test_decoder():
    """Test: Decoder'ı test et."""
    
    print("=" * 70)
    print("[TEST] Decoder")
    print("=" * 70)
    
    # Hyperparameters
    vocab_size = 2000
    embedding_dim = 256
    hidden_dim = 512
    encoder_hidden_dim = 1024  
    num_layers = 2
    dropout = 0.3
    
    batch_size = 2
    src_len = 5
    
    # Test her attention type
    attention_types = ["bahdanau", "luong", "scaled_dot"]
    
    for attention_type in attention_types:
        print(f"\n" + "=" * 70)
        print(f"[INFO] Testing Decoder with {attention_type.upper()} attention")
        print(f"=" * 70)
        
        # Model oluştur
        decoder = Decoder(
            vocab_size, embedding_dim, hidden_dim,
            encoder_hidden_dim, attention_type,
            num_layers, dropout
        )
        decoder.init_weights()
        decoder.eval()
        
        # Parametre sayısı
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"\n   Parameters: {total_params:,}")
        
        # Dummy input
        input_token = torch.randint(0, vocab_size, (batch_size,))
        hidden = torch.randn(num_layers, batch_size, hidden_dim)
        encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_dim)
        
        print(f"\n Input:")
        print(f"   input_token: {input_token.shape}")
        print(f"   hidden: {hidden.shape}")
        print(f"   encoder_outputs: {encoder_outputs.shape}")
        
        # Forward pass
        with torch.no_grad():
            output, new_hidden, attention_weights = decoder(
                input_token, hidden, encoder_outputs
            )
        
        print(f"\n  Output:")
        print(f"   output: {output.shape}")  # [batch_size, vocab_size]
        print(f"   new_hidden: {new_hidden.shape}")  # [num_layers, batch_size, hidden_dim]
        print(f"   attention_weights: {attention_weights.shape}")  # [batch_size, src_len]
        
        # Output analizi
        print(f"\n Output Analysis (batch 0):")
        probs = F.softmax(output[0], dim=-1)
        top5_probs, top5_ids = torch.topk(probs, 5)
        
        print(f"   Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_ids)):
            print(f"      {i+1}. Token {idx.item():4d}: {prob.item():.4f}")
        
        # Attention analizi
        print(f"\n Attention Analysis (batch 0):")
        print(f"   Attention weights: {attention_weights[0].tolist()}")
        print(f"   Sum: {attention_weights[0].sum():.4f} (should be 1.0)")
        print(f"   Max position: {attention_weights[0].argmax().item()}")
        print(f"   Max weight: {attention_weights[0].max():.4f}")
    
    print("\n" + "=" * 70)
    print("[DONE] Decoder test tamamlandı!")
    print("=" * 70)
    
    print("\n Decoder Workflow:")
    print("   1. Kelime → Embedding")
    print("   2. Attention → Context vector")
    print("   3. Embedding + Context → GRU")
    print("   4. GRU output → Vocabulary probabilities")
    print("   5. En yüksek olasılık → Sonraki kelime")


if __name__ == "__main__":
    test_decoder()

