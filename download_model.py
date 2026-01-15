#!/usr/bin/env python3
"""Download model from WandB registry."""
import os
import sys
from pathlib import Path

import wandb

api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("Warning: WANDB_API_KEY not set, skipping model download")
    print("Note: Model file must be provided via COPY or volume mount")
    sys.exit(0)

try:
    wandb.login(key=api_key)
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "mlops-group-85")
    project = os.environ.get("WANDB_PROJECT", "mlops-project")
    artifact_name = os.environ.get("WANDB_ARTIFACT", "best_model:latest")
    
    print(f"Downloading artifact: {entity}/{project}/{artifact_name}")
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
    artifact_dir = artifact.download()
    
    import shutil
    
    model_file = Path(artifact_dir) / "best_model.pt"
    if model_file.exists():
        shutil.copy(model_file, "models/best_model.pt")
        print(f"Model downloaded successfully to models/best_model.pt")
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
