from pathlib import Path

class Config:
    def __init__(self):
        self.model_repo : str = "Majipa/axiom"
        self.model_dir : Path = Path(".local/model")
        self.model_path : Path = self.model_dir / "model.safetensors"
