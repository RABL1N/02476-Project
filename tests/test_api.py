import io
from fastapi.testclient import TestClient
from PIL import Image
import torch
import pytest

import mlops_project.api as api


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """
    Replace the real model with a lightweight dummy model
    so tests do not depend on WandB or downloaded weights.
    """

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Return fixed logits for 2 classes
            batch_size = x.shape[0]
            return torch.tensor([[0.1, 0.9]] * batch_size)

    dummy_model = DummyModel()
    dummy_model.eval()

    monkeypatch.setattr(api, "_model", dummy_model)
    yield


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    # Create a dummy RGB image (224x224)
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in {"NORMAL", "PNEUMONIA"}
