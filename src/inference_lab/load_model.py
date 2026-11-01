# import torch
# import sentencepiece as spm 
from config.config import Config 
from huggingface_hub import snapshot_download

class Load_Model:
    def __init__(self) -> None: 
        self.cnf = Config()        

    def download_model(self):
        model_repo = self.cnf.model_repo
        snapshot_download(repo_id=model_repo,
                          repo_type='model',
                          local_dir=".local/model/")
        print(f"model successfully downloaded from {model_repo}")


