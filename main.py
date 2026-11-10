import torch

from src.inference_lab.load_model import Load_Model
from src.inference_lab.generate import Generator 

model_loader = Load_Model()
model = model_loader.get_model() 
generator = Generator(model= model, context_length=2048)
x = torch.tensor([[1, 40, 3]], device="cuda")
generator.benchmark(x, 512)


