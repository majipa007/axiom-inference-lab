from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, d_model:int) -> None:
        super().__init__()
        self.gelu_stack = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return self.gelu_stack(x)

