import json

import torch
from config.config import Config 
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from src.model.model import Axiom

class Load_Model:
    def __init__(self) -> None: 
        self.cnf = Config()
        with open(".local/model/config.json") as f:
            data = json.load(f)
        self.device = "cuda"
        self.model = Axiom(
            vocab_size= data["vocab_size"], 
            context_length=data["context_length"],
            d_model=data["d_model"],
            n_blocks=data["n_blocks"],
            n_heads=data["n_heads"],
        )

    def download_model(self):
        """
            Simply download your model.
        """
        model_repo = self.cnf.model_repo
        snapshot_download(repo_id=model_repo,
                          repo_type='model',
                          local_dir=".local/model/")
        print(f"model successfully downloaded from {model_repo}")
    
    def model_info(self):
        """
            checking the parameters.
        """
        total = 1
        with safe_open(self.cnf.model_path, framework="pt") as f:
            for key in f.keys():
                if "mask" in key.lower():
                    print(key, f.get_tensor(key).shape)
                    total = total + f.get_tensor(key).shape[0]*f.get_tensor(key).shape[1]
        print(total)

    def load_model_from_code(self):
        state_dict = load_file(
            self.cnf.model_path,
            device=self.device
        )
        self.model.load_state_dict(state_dict, strict = True)
        self.model.to(device=self.device)
        self.model.eval()

    def forward_pass(self):
        self.load_model_from_code()
        x = torch.tensor([[1, 2, 3]], device=self.device)
        with torch.inference_mode():
            logits = self.model(x)
        print("input:", x.shape)
        print("output:", logits.shape)


        



