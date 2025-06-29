import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


if __name__ == "__main__":
    model = PositionalEncoding(d_model=16)
    dummy_input = torch.zeros(1, 10, 16)  # (batch=1, seq_len=10, d_model=16)
    out = model(dummy_input)
    print("Output shape:", out.shape)
    print(out[0, 0, :5])  # Print first token's first 5 dimensions
