import time
from torch import nn, randint
import torch

class Generator:
    def __init__(self, model: nn.Module, context_length: int) -> None:
        self.model = model
        self.context_length = context_length

    def generate(self, x:torch.tensor, max_new_tokens: int, eos_token_id: int):
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_context = x[:, -self.context_length: ]
                logits =  self.model(x_context)
                last_logits = logits[:, -1, :]
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                x = torch.cat((x, next_token), dim=1)
                # if eos_token_id is not None and torch.all(next_token==eos_token_id):
                #     break
        print("elapsed time:", elapsed)
        return x
    
    def benchmark(self, inp_x:torch.tensor, max_new_tokens: int):
        self.model.eval()
        # warming up my cuda cores
        for i in range(5): # 5 warmup runs 
            x = inp_x
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # x = torch.tensor([[randint(0, 8000, (1,)),randint(0, 8000, (1,)),randint(0, 8000, (1,)) ]], device="cuda")
                    x_context = x[:, -self.context_length: ]
                    logits =  self.model(x_context)
                    last_logits = logits[:, -1, :]
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    x = torch.cat((x, next_token), dim=1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        x = inp_x
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_context = x[:, -self.context_length: ]
                logits =  self.model(x_context)
                last_logits = logits[:, -1, :]
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                x = torch.cat((x, next_token), dim=1)
                # if eos_token_id is not None and torch.all(next_token==eos_token_id):
                #     break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        print("elapsed time:", elapsed)
        return x



# with sampling and temerature
    # def generate(self, x:torch.tensor, max_new_tokens: int, eos_token_id: int):
    #     self.model.eval()
    #     with torch.no_grad():
    #         for _ in range(max_new_tokens):
    #             x_context = x[:, -self.context_length: ]
    #             logits =  self.model(x_context)
    #             last_logits = logits[:, -1, :]
    #             last_logits = last_logits/0.8 # adding temperature
    #             top_values, _ = torch.topk(last_logits, k=40)
    #             threshold = top_values[:,-1].unsqueeze(-1)
    #             last_logits = last_logits.masked_fill(last_logits<threshold, float("-inf"))
    #             probabilities = torch.softmax(last_logits, dim = -1)
    #             next_token = torch.multinomial(probabilities, num_samples=1)
    #             x = torch.cat((x, next_token), dim=1)
    #             if eos_token_id is not None and torch.all(next_token==eos_token_id):
    #                 break
    #
    #     return x
