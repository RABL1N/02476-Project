from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

from mlops_project.model import Model

# -------------------------
# App setup
# -------------------------
app = FastAPI(title="Pneumonia Inference API")

# -------------------------
# Model loading
# -------------------------
MODEL_PATH = Path("models/best_model.pt")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Class mapping
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# -------------------------
# Inference transforms
# -------------------------
inference_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")

    # Preprocess
    x = inference_transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "class_index": pred_idx,
    }
