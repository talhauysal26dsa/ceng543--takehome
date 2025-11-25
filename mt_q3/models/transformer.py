import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, config, pad_idx=0):
        super().__init__()
        d_model = config["transformer"]["d_model"]
        nhead = config["transformer"]["nhead"]
        num_enc = config["transformer"]["num_encoder_layers"]
        num_dec = config["transformer"]["num_decoder_layers"]
        dim_ff = config["transformer"]["dim_feedforward"]
        dropout = config["transformer"]["dropout"]
        activation = config["transformer"].get("activation", "relu")

        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def make_src_key_padding_mask(self, src):
        return (src == self.pad_idx)

    def make_tgt_masks(self, tgt):
        batch_size, tgt_len = tgt.shape
        tgt_pad_mask = (tgt == self.pad_idx)
        subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
        tgt_mask = subsequent_mask
        return tgt_mask, tgt_pad_mask

    def forward(self, src, tgt):
        # src: [batch, src_len], tgt: [batch, tgt_len]
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        tgt_mask, tgt_key_padding_mask = self.make_tgt_masks(tgt)

        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(output)
    
    def translate(self, src, src_lengths, max_length=50, sos_idx=1, eos_idx=2):
                self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encode source
            src_key_padding_mask = self.make_src_key_padding_mask(src)
            src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            
            # Start with SOS token
            tgt_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Create target mask
                tgt_mask, tgt_key_padding_mask = self.make_tgt_masks(tgt_tokens)
                
                # Decode
                tgt_emb = self.pos_decoder(self.tgt_embed(tgt_tokens) * math.sqrt(self.d_model))
                output = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                
                # Get next token
                logits = self.generator(output[:, -1, :])  # [batch, vocab]
                next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
                
                # Append to sequence
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                
                # Check if all sequences have generated EOS
                if (next_token == eos_idx).all():
                    break
            
            # Return first sequence (batch_size=1 for evaluation)
            tokens = tgt_tokens[0].tolist()
            return tokens, None  # No attention weights for now
