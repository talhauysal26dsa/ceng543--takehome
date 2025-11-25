

import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    
    
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
    
    def create_mask(self, src):
      
        mask = (src != self.src_pad_idx)
        return mask
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
      
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        # Output tensors
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.size(1)).to(self.device)
        
        # 1. Encode
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # 2. Decoder'ı initialize et
        decoder_hidden = self.decoder.init_hidden(encoder_hidden)
        
        # 3. Mask oluştur
        mask = self.create_mask(src)
        
        # 4. İlk input (<sos> token)
        decoder_input = tgt[:, 0]  # [batch_size] (ilk sütun = <sos>)
        
        # 5. Decode (her kelime için)
        for t in range(1, tgt_len):  # 1'den başla (<sos> hariç)
            # Decoder step
            output, decoder_hidden, attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                mask
            )
            
            # Output'u kaydet
            outputs[:, t, :] = output
            attentions[:, t, :] = attention_weights
            
            # Teacher forcing karar
            teacher_force = random.random() < teacher_forcing_ratio
            
            # En yüksek olasılıklı token
            top1 = output.argmax(dim=-1)  # [batch_size]
            
            # Bir sonraki input
            decoder_input = tgt[:, t] if teacher_force else top1
        
        return outputs, attentions
    
    def translate(self, src, src_lengths, tgt_vocab, max_length=50, 
                  sos_idx=1, eos_idx=2):
        
        self.eval()  # Evaluation mode
        
        with torch.no_grad():
            # 1. Encode
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            
            # 2. Decoder initialize
            decoder_hidden = self.decoder.init_hidden(encoder_hidden)
            
            # 3. Mask
            mask = self.create_mask(src)
            
            # 4. İlk input (<sos>)
            decoder_input = torch.tensor([sos_idx]).to(self.device)
            
            # 5. Decode (greedy)
            tokens = [sos_idx]
            attentions = []
            
            for _ in range(max_length):
                # Decoder step
                output, decoder_hidden, attention_weights = self.decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    mask
                )
                
                # En yüksek olasılıklı token
                pred_token = output.argmax(dim=-1).item()
                
                # Kaydet
                tokens.append(pred_token)
                attentions.append(attention_weights.squeeze(0).cpu())
                
                # <eos> görüldü mü?
                if pred_token == eos_idx:
                    break
                
                # Bir sonraki input
                decoder_input = torch.tensor([pred_token]).to(self.device)
            
            # Attentions'ı stack et
            attentions = torch.stack(attentions, dim=0)  # [tgt_len, src_len]
        
        return tokens, attentions


def build_model(src_vocab_size, tgt_vocab_size, config, device):
    
    # Config'den hyperparameter'ları al
    embedding_dim = config['model']['embedding_dim']
    encoder_hidden_dim = config['model']['encoder_hidden_dim']
    decoder_hidden_dim = config['model']['decoder_hidden_dim']
    encoder_layers = config['model']['encoder_layers']
    decoder_layers = config['model']['decoder_layers']
    encoder_dropout = config['model']['encoder_dropout']
    decoder_dropout = config['model']['decoder_dropout']
    
    # Encoder
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=encoder_hidden_dim,
        num_layers=encoder_layers,
        dropout=encoder_dropout
    )
    
    # Bidirectional encoder → hidden_dim * 2
    encoder_output_dim = encoder_hidden_dim * 2
    
    
    
    return encoder, encoder_output_dim


def create_seq2seq_model(src_vocab_size, tgt_vocab_size, attention_type,
                         config, device):
   
    # Config
    embedding_dim = config['model']['embedding_dim']
    encoder_hidden_dim = config['model']['encoder_hidden_dim']
    decoder_hidden_dim = config['model']['decoder_hidden_dim']
    encoder_layers = config['model']['encoder_layers']
    decoder_layers = config['model']['decoder_layers']
    encoder_dropout = config['model']['encoder_dropout']
    decoder_dropout = config['model']['decoder_dropout']
    
    # Encoder
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=encoder_hidden_dim,
        num_layers=encoder_layers,
        dropout=encoder_dropout
    )
    encoder.init_weights()
    
    # Encoder output dim (bidirectional)
    encoder_output_dim = encoder_hidden_dim * 2
    
    # Decoder
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=decoder_hidden_dim,
        encoder_hidden_dim=encoder_output_dim,
        attention_type=attention_type,
        num_layers=decoder_layers,
        dropout=decoder_dropout
    )
    decoder.init_weights()
    
    # Seq2Seq
    src_pad_idx = 0  # <pad> token ID
    tgt_pad_idx = 0
    
    model = Seq2Seq(encoder, decoder, src_pad_idx, tgt_pad_idx, device)
    model = model.to(device)
    
    # Toplam parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[INFO] Model olusturuldu: {attention_type.upper()} Attention")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def test_seq2seq():
    """Test: Seq2Seq model'i test et."""
    
    print("=" * 70)
    print("[TEST] Seq2Seq Model")
    print("=" * 70)
    
    # Dummy config
    config = {
        'model': {
            'embedding_dim': 256,
            'encoder_hidden_dim': 512,
            'decoder_hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'encoder_dropout': 0.3,
            'decoder_dropout': 0.3
        }
    }
    
    # Params
    src_vocab_size = 1000
    tgt_vocab_size = 2000
    device = torch.device('cpu')
    
    # Test her attention type
    attention_types = ["bahdanau", "luong", "scaled_dot"]
    
    for attention_type in attention_types:
        print(f"\n" + "=" * 70)
        print(f"[INFO] Testing {attention_type.upper()} Attention")
        print(f"=" * 70)
        
        # Model oluştur
        model = create_seq2seq_model(
            src_vocab_size, tgt_vocab_size,
            attention_type, config, device
        )
        
        # Dummy data
        batch_size = 2
        src_len = 5
        tgt_len = 6
        
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))
        src_lengths = torch.tensor([5, 3])
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
        
        print("\n[INFO] Input:")
        print(f"   src: {src.shape}")
        print(f"   tgt: {tgt.shape}")
        
        # Forward pass (training)
        model.train()
        outputs, attentions = model(src, src_lengths, tgt, teacher_forcing_ratio=0.5)
        
        print("\n[INFO] Output (Training):")
        print(f"   outputs: {outputs.shape}")  # [batch, tgt_len, vocab_size]
        print(f"   attentions: {attentions.shape}")  # [batch, tgt_len, src_len]
        
        # Inference test
        print("\n[INFO] Testing Inference...")
        src_single = src[0:1]  # İlk örnek
        src_lengths_single = src_lengths[0:1]
        
        tokens, attention_weights = model.translate(
            src_single, src_lengths_single,
            tgt_vocab=None,
            max_length=10,
            sos_idx=1,
            eos_idx=2
        )
        
        print(f"   Generated tokens: {tokens}")
        print(f"   Attention shape: {attention_weights.shape}")
    
    print("\n" + "=" * 70)
    print("[DONE] Seq2Seq model test tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    test_seq2seq()

