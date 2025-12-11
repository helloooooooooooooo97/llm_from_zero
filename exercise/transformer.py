import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        # Self-attention + Add & Norm
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # Feed Forward + Add & Norm
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked Self-attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        # Cross-attention
        attn_output, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        # Feed Forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [B, S], tgt: [B, T]
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()
        device = src.device

        src_pos = torch.arange(0, src_len, device=device).unsqueeze(0).expand(batch_size, src_len)
        tgt_pos = torch.arange(0, tgt_len, device=device).unsqueeze(0).expand(batch_size, tgt_len)

        src_emb = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(tgt_pos)

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)

        logits = self.out_proj(dec_out)  # [B, T, vocab_size]
        return logits

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. Mask out subsequent positions (for decoder).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)