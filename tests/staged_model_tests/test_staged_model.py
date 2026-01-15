import wandb
import os
import time
import torch
from src.mlops_project.model import Model as MyModel

def load_model(artifact):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    logdir = "./artifacts"
    
    artifact = api.artifact(artifact)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    # Standard PyTorch loading pattern
    # You may need to adjust num_classes if your model was saved with a different value
    model = MyModel(num_classes=2)
    model.load_state_dict(torch.load(f"{logdir}/{file_name}", map_location="cpu"))
    model.eval()
    return model

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1