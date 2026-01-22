import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def get_loaded_model():
    model_name = os.getenv("MODEL_NAME")
    assert model_name, "MODEL_NAME environment variable must be set."
    return load_model(model_name)


def test_model_speed():
    model = get_loaded_model()
    start = time.time()
    for _ in range(10):
        # Use correct input shape for your model
        model(torch.rand(1, 3, 224, 224))
    end = time.time()
    assert end - start < 2


def test_model_forward_shape():
    model = get_loaded_model()
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert output.shape == (4, 2)


def test_model_output_dtype():
    model = get_loaded_model()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.dtype == torch.float32


def test_model_output_is_finite():
    model = get_loaded_model()
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


def test_model_has_trainable_parameters():
    model = get_loaded_model()
    params = list(model.parameters())
    assert len(params) > 0, "Model has no parameters"
    assert all(p.requires_grad for p in params), "Not all parameters are trainable"
