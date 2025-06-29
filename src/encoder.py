
import torch
from src.attention import MultiHeadAttention

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model),
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x2 = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(x2 + self.ff(x2))
