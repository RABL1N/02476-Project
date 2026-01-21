#!/usr/bin/env python3
"""Download model from WandB registry."""
import os
import sys
from pathlib import Path

import wandb

api_key = os.environ.get("WANDB_API_KEY")
try:
    # Try to login - if API key is provided, use it; otherwise use stored credentials
    if api_key:
        wandb.login(key=api_key)
    else:
        # If no API key in env, try using stored credentials from 'wandb login'
        wandb.login()
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "mlops-group-85")
    project = os.environ.get("WANDB_PROJECT", "mlops-project")
    artifact_name = os.environ.get("WANDB_ARTIFACT", "best_model:best")

    # Use the specified artifact (defaults to "best_model:best" which uses the "best" alias)
    print(f"Downloading artifact: {entity}/{project}/{artifact_name}")
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}")

    artifact_dir = artifact.download()

    import shutil

    model_file = Path(artifact_dir) / "best_model.pt"
    if model_file.exists():
        shutil.copy(model_file, "models/best_model.pt")

        # Save artifact metadata to a file for the API to read
        metadata_file = Path("models/artifact_metadata.json")
        import json

        metadata_info = {
            "artifact_name": artifact.name,
            "artifact_version": artifact.version,
            "artifact_id": artifact.id,
            "validation_accuracy": artifact.metadata.get("best_val_accuracy", None),
            "best_val_loss": artifact.metadata.get("best_val_loss", None),
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata_info, f, indent=2)

        print("Model downloaded successfully to models/best_model.pt")
        print(
            f"Model version: {artifact.version}, Validation accuracy: {artifact.metadata.get('best_val_accuracy', 'N/A')}%"
        )
    else:
        print(f"Error: best_model.pt not found in artifact directory: {artifact_dir}")
        # List what's actually in the artifact
        print(f"Files in artifact: {list(Path(artifact_dir).iterdir())}")
        sys.exit(1)
except Exception as e:
    print(f"Error downloading model from WandB: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
