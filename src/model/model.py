import torch
from torch import nn
from src.model.block import TransformerBlock

class Axiom(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # V no of items in the vocab
        context_length: int,  # T in BTC this is the time dimension
        d_model: int,  # C, how many dimensions each token will have
        n_heads: int,  # No of attention heads
        n_blocks: int, # No of TransformerBlocks
    ) -> None:
        super().__init__()

        self.context_length = context_length

        # [V, C]
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # [T, C]
        self.position_embedding = nn.Embedding(context_length, d_model)

        # [C, V]
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                context_length=context_length,
            )
            for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(d_model) 

    def forward(self, x:torch.Tensor)->torch.Tensor:
        token_embeddings = self.token_embedding(x)
        positions = torch.arange(x.shape[1], device=x.device) # tokens in the current input, they might or might not be max tokens
        position_embedding = self.position_embedding(positions)
        x = token_embeddings+position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
        
