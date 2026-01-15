from fastapi.testclient import TestClient
from PIL import Image
import io

from mlops_project.api import app

client = TestClient(app)


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
