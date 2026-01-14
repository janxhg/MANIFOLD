
import torch
import torch.nn as nn
import math

class MicroGPT(nn.Module):
    """
    A lightweight GPT implementation (Decoder-only Transformer) 
    designed to match the parameter count of GFN-Medium for fair comparison.
    """
    def __init__(self, vocab_size, dim, depth, heads, max_len=10000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(0.1)
        
        # Transformer Block
        # norm_first=True (Pre-LN) is standard for GPT-2/3 stability
        layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=4*dim, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        b, t = idx.size()
        if t > self.pos_emb.size(1):
            raise ValueError(f"Sequence length {t} exceeds max length {self.pos_emb.size(1)}")
            
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        
        # Causal mask (ensure autoregressive property)
        # 0 on diagonal and below, -inf above
        mask = torch.triu(torch.ones(t, t, device=idx.device) * float('-inf'), diagonal=1)
        
        # is_causal=True is optimized in PyTorch 2.0+
        x = self.blocks(x, mask=mask, is_causal=True)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
