import torch
from torch import nn
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import MLP

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 context_length:int,
                 ) -> None:
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.multiHeadAttention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, context_length=context_length)
        self.mlp = MLP(d_model=d_model)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x+self.multiHeadAttention(self.layerNorm1(x))
        x = x+self.mlp(self.layerNorm2(x))
        return x
        


    
