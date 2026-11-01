import torch 
from torch import nn 

class SingleHeadAttention(nn.Module):
    def __init__(self, 
                 d_model:int, # dimention of the model's training vectors
                 head_size: int, # head size of the single head attnetion 
                 context_length: int, # context lenght or the  max lenght of tokens we take
                 ) -> None:
        super().__init__()
        self.head_size = head_size
        self.context_length = context_length
        self.query = nn.Linear(
            d_model, 
            head_size, 
            bias=False
        )

        self.key = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool
                )
            ),
            # persistent=False # this is so that the model doesn't add causal mask as trainable param count
        )

    

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x = [B, T, C]
        _, sequence_length, _ = x.shape
        causal_mask: torch.Tensor = self.get_buffer("causal_mask")

        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence_length exeeds: {sequence_length}"
            ) 

        # [B, T, C] gets converted to [B, T, head_size]
        # query = x.WQT
        # x : [B T C] | WQT: [C H]
        # query = [B T H]
        query = self.query(x) 
        key = self.key(x)
        value = self.value(x)


        # [B T H] @ [B H T] ->  [B T T]
        # scores = Query@Key(Transpose)
        scores = query @ key.transpose(-2, -1)

        # scaling the score with root of the head size
        scaled_scores = scores/(self.head_size ** 0.5 )

        ## [T, T]
        mask = causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        # future tokens are -inifity so that softmax returns 0
        masked_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = torch.softmax(masked_scores, dim=-1)

        output = attention_weights @ value

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int, # dimention of the model 
                 n_heads: int, # no of single head attention
                 context_length: int, # max length 
                 ) -> None:
        super().__init__()

        # 1. make the d_model divisible by n_heads
        if d_model%n_heads!=0:
            raise ValueError(
                f"d_model : {d_model} is not divisible by n_heads {n_heads}"
            )

        # 2. calculate the head_size ( head_size = d_model // n_heads )
        self.head_size = d_model//n_heads

        # 3. create n_heads SingleHeadAttention modules using nn.ModuleList
        # using a module list because pytorch registers all those attention heads as parts of your model 
        
        self.heads = nn.ModuleList(
            SingleHeadAttention(
                d_model=d_model,
                head_size=self.head_size,
                context_length=context_length
            )
            for _ in range(n_heads)
        )
        
        # 4. create the output projection -> Linear(c, c)
        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False
        )

    def forward(self, x: torch.Tensor)->torch.Tensor:
        results = []
        for head in self.heads:
            result = head(x)
            results.append(result)
        concatinated_result = torch.cat(results, dim = -1)
        output = self.output_projection(concatinated_result)
        return output 
