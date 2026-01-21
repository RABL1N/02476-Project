from pathlib import Path
import io
from datetime import datetime
import json

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms

from mlops_project.model import Model

app = FastAPI()

# Configuration
MODEL_PATH = Path("models/best_model.pt")
DEVICE = "cpu"

# Class mapping
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

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

    model = Model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    _model = model
    return model


# Image preprocessing (same as training)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    """Get information about the loaded model."""
    model_stat = MODEL_PATH.stat() if MODEL_PATH.exists() else None
    metadata_path = Path("models/artifact_metadata.json")

    # Load model to get info (if not already loaded)
    model = load_model() if MODEL_PATH.exists() else None

    info = {
        "model_path": str(MODEL_PATH.absolute()),
        "model_exists": MODEL_PATH.exists(),
        "device": DEVICE,
        "model_architecture": "CNN (ChestXRay)",
        "num_classes": 2,
        "class_names": CLASS_NAMES,
    }

    if model_stat:
        info.update(
            {
                "model_size_bytes": model_stat.st_size,
                "model_size_mb": round(model_stat.st_size / (1024 * 1024), 2),
                "model_modified": datetime.fromtimestamp(
                    model_stat.st_mtime
                ).isoformat(),
            }
        )

    # Count model parameters
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info.update(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
        )

    # Add WandB artifact information if available
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                artifact_metadata = json.load(f)
            info["wandb_artifact"] = {
                "name": artifact_metadata.get("artifact_name"),
                "version": artifact_metadata.get("artifact_version"),
                "validation_accuracy": artifact_metadata.get("validation_accuracy"),
                "best_val_loss": artifact_metadata.get("best_val_loss"),
            }
        except Exception as e:
            info["wandb_artifact"] = {"error": f"Could not read metadata: {e}"}
    else:
        info["wandb_artifact"] = {
            "note": "Metadata not available (model may not be from WandB)"
        }

    return info


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
        x = preprocess(image).unsqueeze(0).to(DEVICE)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run inference
    with torch.no_grad():
        outputs = model(x)
        predicted_class = torch.argmax(outputs, dim=1).item()

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}
    return {"prediction": label_map[predicted_class], "class_index": predicted_class}
