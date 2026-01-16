import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import time
import torch
from src.mlops_project.model import Model as MyModel

def load_model(artifact):
    api_key = os.getenv("WANDB_API_KEY")
    print(api_key)
    if not api_key:
        raise RuntimeError("WANDB_API_KEY environment variable is not set. Please set it before running the test.")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT environment variables must be set.")

    api = wandb.Api(api_key=api_key, overrides={"entity": entity, "project": project})
    logdir = "./artifacts"

    artifact_obj = api.artifact(artifact)
    artifact_obj.download(root=logdir)
    file_name = artifact_obj.files()[0].name
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