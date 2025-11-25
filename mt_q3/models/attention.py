import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
        def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: [1, batch, dec_hidden]  (top layer)
        # encoder_outputs: [batch, src_len, enc_hidden*2]
        batch_size, src_len, _ = encoder_outputs.shape
        dec_hidden = decoder_hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(-1)  # [batch, src_len]
        
        # Adjust mask size to match encoder_outputs if needed
        if mask.size(1) != src_len:
            # Truncate or pad mask to match encoder_outputs length
            mask = mask[:, :src_len]
        
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
