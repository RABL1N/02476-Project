from pathlib import Path
import io

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from mlops_project.model import Model



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for course/demo
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = Path("models/best_model.pt")
DEVICE = "cpu"

# Global model cache (lazy-loaded)
_model = None


def load_model() -> Model:
    """
    Lazily load the model.
    This function is ONLY called when /predict is hit.
    """
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    _model = model
    return model


# Image preprocessing (same as training)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Load model lazily
    try:
        model = load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Read and preprocess image
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image).unsqueeze(0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference
    with torch.no_grad():
        outputs = model(x)
        predicted_class = torch.argmax(outputs, dim=1).item()

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}
    return {"prediction": label_map[predicted_class]}
