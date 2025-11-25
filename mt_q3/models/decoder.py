import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AdditiveAttention

class Decoder(nn.Module):
        def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        encoder_hidden_dim,
        num_layers=2,
        dropout=0.3,
        attention_dim=512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = AdditiveAttention(encoder_hidden_dim, hidden_dim, attention_dim)
        self.gru = nn.GRU(
            embedding_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.xavier_uniform_(self.fc_out.weight.data)
        nn.init.zeros_(self.fc_out.bias.data)

    def init_hidden(self, encoder_hidden):
        # encoder_hidden: [num_enc_layers*2, batch, enc_hidden] (bidirectional)
        num_enc_layers = encoder_hidden.size(0) // 2
        batch_size = encoder_hidden.size(1)
        enc_hidden_dim = encoder_hidden.size(2)

        # reshape to [num_enc_layers, directions=2, batch, hidden] then sum directions
        encoder_hidden = encoder_hidden.view(num_enc_layers, 2, batch_size, enc_hidden_dim)
        combined = encoder_hidden.sum(dim=1)  # [num_enc_layers, batch, hidden]

        # adjust if decoder layers differ from encoder layers
        dec_layers = self.gru.num_layers
        if combined.size(0) != dec_layers:
            if dec_layers < combined.size(0):
                combined = combined[:dec_layers]
            else:
                # repeat last layer to match required decoder layers
                repeat_needed = dec_layers - combined.size(0)
                pad_layers = combined[-1:].repeat(repeat_needed, 1, 1)
                combined = torch.cat([combined, pad_layers], dim=0)

        return combined

    def forward(self, input_token, hidden, encoder_outputs, mask):
        # input_token: [batch]
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)  # [batch, 1, emb]
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)  # context: [batch, enc_hidden*2]
        gru_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [batch,1, emb+enc]
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)  # [batch, dec_hidden]
        output_logits = self.fc_out(torch.cat((output, context, embedded.squeeze(1)), dim=1))
        return output_logits, hidden, attn_weights
