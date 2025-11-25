import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def create_mask(self, src):
        return (src != self.src_pad_idx).to(self.device)

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size(0), tgt.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.size(1), device=self.device)

        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        decoder_hidden = self.decoder.init_hidden(encoder_hidden)
        mask = self.create_mask(src)
        decoder_input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, decoder_hidden, attention_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            outputs[:, t, :] = output
            attentions[:, t, :] = attention_weights
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=-1)
            decoder_input = tgt[:, t] if teacher_force else top1

        return outputs, attentions

    def translate(self, src, src_lengths, max_length=50, sos_idx=1, eos_idx=2):
        self.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            decoder_hidden = self.decoder.init_hidden(encoder_hidden)
            mask = self.create_mask(src)
            decoder_input = torch.tensor([sos_idx], device=self.device)
            tokens = [sos_idx]
            attentions = []
            for _ in range(max_length):
                output, decoder_hidden, attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
                pred = output.argmax(dim=-1).item()
                tokens.append(pred)
                attentions.append(attn.squeeze(0).cpu())
                if pred == eos_idx:
                    break
                decoder_input = torch.tensor([pred], device=self.device)
            if attentions:
                attentions = torch.stack(attentions, dim=0)
        return tokens, attentions

def create_seq2seq_model(src_vocab_size, tgt_vocab_size, config, device, src_pad_idx=0, tgt_pad_idx=0):
    embedding_dim = config["seq2seq"]["embedding_dim"]
    enc_hidden = config["seq2seq"]["encoder_hidden_dim"]
    dec_hidden = config["seq2seq"]["decoder_hidden_dim"]
    enc_layers = config["seq2seq"]["encoder_layers"]
    dec_layers = config["seq2seq"]["decoder_layers"]
    enc_dropout = config["seq2seq"]["encoder_dropout"]
    dec_dropout = config["seq2seq"]["decoder_dropout"]
    attn_dim = config["seq2seq"]["attention_dim"]

    encoder = Encoder(vocab_size=src_vocab_size, embedding_dim=embedding_dim, hidden_dim=enc_hidden, num_layers=enc_layers, dropout=enc_dropout)
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=dec_hidden,
        encoder_hidden_dim=enc_hidden * 2,
        num_layers=dec_layers,
        dropout=dec_dropout,
        attention_dim=attn_dim,
    )
    encoder.init_weights()
    decoder.init_weights()

    model = Seq2Seq(encoder, decoder, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx, device=device).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Seq2Seq (Additive) created | params: {total_params:,} (trainable: {trainable_params:,})")
    return model
