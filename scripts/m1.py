import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 100):
        super().__init__()  # new version of: super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        print("Shape of pe:", pe.size())
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: [max_len, 1], Arange: Returns a 1-D tensor from start to stop, Unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position
        print("Shape of position:", position.size())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # Shape: [d_model // 2]
        print("Shape of div_term", div_term.size())
        
        # Expand div_term to match the shape of position
        div_term = div_term.unsqueeze(0)  # Shape: [1, d_model // 2]
        div_term = div_term.expand(max_len, -1)  # Shape: [max_len, d_model // 2]

        # Make sure div_term is of shape [max_len, d_model] to broadcast properly
        div_term_full = torch.zeros(max_len, d_model)
        div_term_full[:, 0::2] = div_term  # Fill every other column with div_term
        print("Corrected shape of div_term", div_term.size())

        pe[:, 0::2] = torch.sin(position * div_term_full[:, 0::2])  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term_full[:, 1::2])  # Cosine for odd indices
        # pe = pe.unsqueeze(0)
        # x = x + pe[:, :x.size(1)]
        
        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        
    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x
    
    
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, d_model=32, nhead=4, dim_feedforward=128, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32))
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output