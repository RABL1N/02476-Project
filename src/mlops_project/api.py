from pathlib import Path
import io
from datetime import datetime
import json

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from mlops_project.model import Model

from pydantic import BaseModel, Field
from typing import List, Optional

import pandas as pd
from evidently.legacy.report import Report as DriftReport
from evidently.legacy.metric_preset import DataDriftPreset



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


# --- Drift detection configuration ---
DRIFT_REF_PATH = (
    Path(__file__).parent / "monitoring" / "artifacts" / "reference_features.csv"
)
_drift_ref_df: Optional[pd.DataFrame] = None


def load_drift_reference() -> pd.DataFrame:
    """Lazy-load reference feature distribution for drift detection."""
    global _drift_ref_df
    if _drift_ref_df is not None:
        return _drift_ref_df

    if not DRIFT_REF_PATH.exists():
        raise RuntimeError(f"Reference drift file not found at {DRIFT_REF_PATH}")

    df = pd.read_csv(DRIFT_REF_PATH)
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    _drift_ref_df = df
    return df


class FeatureRow(BaseModel):
    mean_intensity: float = Field(..., example=0.42)
    std_intensity: float = Field(..., example=0.18)
    min_intensity: float = Field(..., example=0.0)
    max_intensity: float = Field(..., example=0.97)


class DriftFeaturesRequest(BaseModel):
    rows: List[FeatureRow]



@app.post("/drift/features")
def drift_features(req: DriftFeaturesRequest):
    # Load reference data
    try:
        reference_df = load_drift_reference()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Current data from request
    current_df = pd.DataFrame([r.model_dump() for r in req.rows])

    # Run Evidently drift detection (legacy API in Evidently 0.7.x)
    report = DriftReport(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    rep = report.as_dict()
    # Extract summary (defensive across minor schema differences)
    result = rep.get("metrics", [{}])[0].get("result", {})

    return {
        "dataset_drift": result.get("dataset_drift"),
        "number_of_drifted_columns": result.get("number_of_drifted_columns"),
        "number_of_columns": result.get("number_of_columns"),
        "share_of_drifted_columns": (
            result.get("share_of_drifted_columns")
            or result.get("drift_share")
            or result.get("share_drifted_columns")
        ),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "reference_path": str(DRIFT_REF_PATH),
    }

