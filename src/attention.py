import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.size()
        q = self.q_linear(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(x)
