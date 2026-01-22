import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mlops_project.data import ChestXRayDataset
from mlops_project.model_lightning import LitModel

log = logging.getLogger(__name__)
from pytorch_lightning import Trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the CNN model on the chest X-ray pneumonia dataset.

    Args:
        cfg: Hydra configuration object
    """
    # Get original working directory (Hydra changes it to outputs/)
    original_cwd = Path(get_original_cwd())

    # Extract configuration values and resolve relative to original cwd
    data_dir = original_cwd / cfg.data_dir
    model_dir = original_cwd / cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs

    # Extract entity and get Hydra trainer config
    entity = cfg.wandb.get("entity", "mlops-group-85")
    project = cfg.wandb.get("project", "mlops-project")

    # Use PyTorch Lightning Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    wandb_logger = WandbLogger(project=project, entity=entity, log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="best_model",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=getattr(cfg, "early_stopping_patience", 5),
        mode="max",
        verbose=True,
    )

    # Use Hydra configuration for trainer parameters
    accelerator = cfg.trainer.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu")
    devices = cfg.trainer.get("devices", 1)

    trainer = Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        default_root_dir=str(model_dir),
    )

    # Load data
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ChestXRayDataset(data_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(data_dir, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = LitModel(learning_rate=cfg.learning_rate)

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save best model path for artifact logging
    best_model_path = checkpoint_callback.best_model_path
    log.info(f"Best model saved to: {best_model_path}")

    best_val_acc = checkpoint_callback.best_model_score
    best_val_loss = getattr(checkpoint_callback, "best_model_loss", 0.0)

    log.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    log.info(f"Best model saved to: {model_dir / 'best_model.pt'}")

    # Log best model as a wandb artifact
    artifact = wandb.Artifact(
        name="best_model",
        type="model",
        description="Best model based on validation accuracy",
        metadata={"best_val_accuracy": best_val_acc, "best_val_loss": best_val_loss},
    )
    artifact.add_file(str(best_model_path))

    # Get API instance and entity/project info for registry comparison
    # Ensure API is initialized with the correct entity

    # Initialize API - ensure it uses the same authentication as the run
    api = wandb.Api()

    # Determine if this model should get the "best" alias
    # Compare against current "best" model using validation accuracy
    should_be_best = False

    try:
        # Try to get the current "best" model from project artifacts
        try:
            current_best = api.artifact(f"{entity}/{project}/best_model:best")
            current_best_acc = current_best.metadata.get("best_val_accuracy", -1.0) if current_best.metadata else -1.0

            # Compare: higher validation accuracy is better
            # If equal, also promote if validation loss is better (lower)
            if best_val_acc > current_best_acc:
                should_be_best = True
                log.info(
                    f"New model is better: validation accuracy {best_val_acc:.2f}% "
                    f"(previous best: {current_best_acc:.2f}%)"
                )
            elif best_val_acc == current_best_acc:
                # If accuracy is equal, check validation loss (lower is better)
                current_best_loss = (
                    current_best.metadata.get("best_val_loss", float("inf")) if current_best.metadata else float("inf")
                )
                if best_val_loss < current_best_loss:
                    should_be_best = True
                    log.info(
                        f"New model has same accuracy {best_val_acc:.2f}% but better loss "
                        f"({best_val_loss:.4f} vs {current_best_loss:.4f}), promoting to 'best'"
                    )
                else:
                    log.info(
                        f"Current model accuracy {best_val_acc:.2f}% equals existing best "
                        f"and loss is not better, keeping existing 'best' alias"
                    )
            else:
                log.info(
                    f"Current model accuracy {best_val_acc:.2f}% is not better than "
                    f"existing best {current_best_acc:.2f}%, keeping existing 'best' alias"
                )
        except Exception as e:
            # No existing "best" model - this is the first one
            # Check if it's actually a "not found" error or something else
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                should_be_best = True
                log.info("No existing 'best' model found. This will be the first 'best' model.")
            else:
                # Some other error - log it but still try to set as best
                log.warning(f"Error checking existing 'best' model: {e}. Assuming this is the first model.")
                should_be_best = True
    except Exception as e:
        log.warning(f"Could not check existing 'best' model: {e}. Assuming this is the first model.")
        should_be_best = True

    # Set aliases: always "latest", and "best" if this model is better
    # W&B automatically handles alias "theft" - assigning "best" to new version removes it from old one
    aliases = ["latest"]
    if should_be_best:
        aliases.append("best")

    # Log artifact with aliases
    logged_artifact = wandb.log_artifact(artifact, aliases=aliases)
    log.info(f"Model uploaded to WandB as an artifact with aliases: {aliases}")

    # Link to registry immediately after logging (before waiting)
    # This ensures the run context is still active
    registry_name = "02476_registry"
    registry_path = f"wandb-registry-{registry_name}/Models"

    try:
        # Try linking immediately while run is active
        wandb.run.link_artifact(artifact=logged_artifact, target_path=registry_path)
        log.info(f"Artifact linked to registry: {registry_path} with aliases: {aliases}")
    except Exception as e1:
        # If immediate linking fails, wait for artifact to finalize and try API method
        log.warning(f"Immediate linking failed: {e1}")
        log.info("Waiting for artifact to finalize, then trying API method...")
        logged_artifact.wait()

        try:
            # Use API method after artifact is finalized
            artifact_path = f"{entity}/{project}/{logged_artifact.name}:{logged_artifact.version}"
            log.info(f"Attempting to link artifact via API: {artifact_path} to {registry_path}")
            api_artifact = api.artifact(artifact_path)
            api_artifact.link(target_path=registry_path)
            log.info(f"Artifact linked to registry via API: {registry_path} with aliases: {aliases}")
        except Exception as e2:
            log.error(f"Failed to link artifact to registry: {e2}")
            log.error(f"Error type: {type(e2).__name__}")
            log.error(f"Error details: {str(e2)}")
            log.info(f"Artifact is still available in project with aliases: {aliases}")
            log.info(f"Artifact path: {artifact_path if 'artifact_path' in locals() else 'N/A'}")
            log.info("")
            log.info("TROUBLESHOOTING:")
            log.info(f"1. Verify the registry '{registry_name}' exists and you have 'Member' or 'Admin' role")
            log.info("2. Check registry permissions: https://wandb.ai/mlops-group-85/registries/02476_registry")
            log.info("3. Ensure you're logged in: wandb login")
            log.info("4. Verify entity matches: entity should be 'mlops-group-85'")
            log.info("5. You can manually link artifacts in the WandB UI if needed")

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    train()
