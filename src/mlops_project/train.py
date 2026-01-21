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
from mlops_project.model import Model

log = logging.getLogger(__name__)


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

    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs

    learning_rate = cfg.learning_rate
    num_classes = cfg.num_classes

    # Set device
    if cfg.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = cfg.device

    # Setup logging
    log.info("Starting training with configuration:")
    log.info(f"  Data directory: {data_dir}")
    log.info(f"  Model directory: {model_dir}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Number of epochs: {num_epochs}")
    log.info(f"  Learning rate: {learning_rate}")
    log.info(f"  Number of classes: {num_classes}")
    log.info(f"  Device: {device}")

    # Initialize WandB
    # Ensure entity is set to team (required for registry linking)
    entity = cfg.wandb.entity if cfg.wandb.entity else "mlops-group-85"
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        entity=entity,
        name=getattr(cfg.wandb, "name", None),
        tags=getattr(cfg.wandb, "tags", []),
        notes=getattr(cfg.wandb, "notes", None),
        config=wandb_config,
    )
    log.info(f"WandB initialized: {wandb.run.url} (entity: {entity})")

    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Define transforms for training (with data augmentation) and validation
    train_transform = transforms.Compose(
        [
            transforms.Resize(tuple(cfg.augmentation.train.resize)),
            transforms.RandomHorizontalFlip(
                p=cfg.augmentation.train.random_horizontal_flip
            ),
            transforms.RandomRotation(degrees=cfg.augmentation.train.random_rotation),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(tuple(cfg.augmentation.val.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalize.mean, std=cfg.normalize.std),
        ]
    )

    # Create datasets
    train_dataset = ChestXRayDataset(data_dir, split="train", transform=train_transform)
    val_dataset = ChestXRayDataset(data_dir, split="val", transform=val_transform)

    # Override for fast/fake training
    if getattr(cfg, "fake_training", False):
        # Use only a few samples for speed
        train_dataset.image_paths = train_dataset.image_paths[:4]
        train_dataset.labels = train_dataset.labels[:4]
        val_dataset.image_paths = val_dataset.image_paths[:2]
        val_dataset.labels = val_dataset.labels[:2]
        batch_size = 1 * len(train_dataset)
        num_epochs = 1
        log.info(
            "FAKE TRAINING MODE: Using 4 train and 2 val samples, batch_size=4, num_epochs=1"
        )
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    log.info(f"Train dataset size: {len(train_dataset)}")
    log.info(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model
    model = Model(num_classes=num_classes)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model has {num_params:,} parameters")

    # Log model architecture to WandB
    wandb.config.update({"model_params": num_params})

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    patience = getattr(cfg, "early_stopping_patience", 5)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100 * train_correct / train_total
            train_pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"}
            )

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
        )
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                current_acc = 100 * val_correct / val_total
                val_pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"}
                )

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Log metrics to WandB
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/accuracy": train_acc,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
            }
        )

        # Log progress
        log.info(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model and reset patience counter
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            model_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            log.info(f"  -> Saved best model with validation accuracy: {val_acc:.2f}%")
            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_val_loss"] = best_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping check
        if epochs_since_improvement >= patience:
            log.info(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    log.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    log.info(f"Best model saved to: {model_dir / 'best_model.pt'}")

    # Log best model as a wandb artifact
    artifact = wandb.Artifact(
        name="best_model",
        type="model",
        description="Best model based on validation accuracy",
        metadata={"best_val_accuracy": best_val_acc, "best_val_loss": best_val_loss},
    )
    artifact.add_file(str(model_dir / "best_model.pt"))

    # Get API instance and entity/project info for registry comparison
    # Ensure API is initialized with the correct entity
    entity = wandb.run.entity or "mlops-group-85"
    project = wandb.run.project or "mlops-project"

    # Initialize API - ensure it uses the same authentication as the run
    api = wandb.Api()

    # Determine if this model should get the "best" alias
    # Compare against current "best" model using validation accuracy
    should_be_best = False

    try:
        # Try to get the current "best" model from project artifacts
        try:
            current_best = api.artifact(f"{entity}/{project}/best_model:best")
            current_best_acc = (
                current_best.metadata.get("best_val_accuracy", -1.0)
                if current_best.metadata
                else -1.0
            )

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
                    current_best.metadata.get("best_val_loss", float("inf"))
                    if current_best.metadata
                    else float("inf")
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
                log.info(
                    "No existing 'best' model found. This will be the first 'best' model."
                )
            else:
                # Some other error - log it but still try to set as best
                log.warning(
                    f"Error checking existing 'best' model: {e}. Assuming this is the first model."
                )
                should_be_best = True
    except Exception as e:
        log.warning(
            f"Could not check existing 'best' model: {e}. Assuming this is the first model."
        )
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
        log.info(
            f"Artifact linked to registry: {registry_path} with aliases: {aliases}"
        )
    except Exception as e1:
        # If immediate linking fails, wait for artifact to finalize and try API method
        log.warning(f"Immediate linking failed: {e1}")
        log.info("Waiting for artifact to finalize, then trying API method...")
        logged_artifact.wait()

        try:
            # Use API method after artifact is finalized
            artifact_path = (
                f"{entity}/{project}/{logged_artifact.name}:{logged_artifact.version}"
            )
            log.info(
                f"Attempting to link artifact via API: {artifact_path} to {registry_path}"
            )
            api_artifact = api.artifact(artifact_path)
            api_artifact.link(target_path=registry_path)
            log.info(
                f"Artifact linked to registry via API: {registry_path} with aliases: {aliases}"
            )
        except Exception as e2:
            log.error(f"Failed to link artifact to registry: {e2}")
            log.error(f"Error type: {type(e2).__name__}")
            log.error(f"Error details: {str(e2)}")
            log.info(f"Artifact is still available in project with aliases: {aliases}")
            log.info(
                f"Artifact path: {artifact_path if 'artifact_path' in locals() else 'N/A'}"
            )
            log.info("")
            log.info("TROUBLESHOOTING:")
            log.info(
                f"1. Verify the registry '{registry_name}' exists and you have 'Member' or 'Admin' role"
            )
            log.info(
                "2. Check registry permissions: https://wandb.ai/mlops-group-85/registries/02476_registry"
            )
            log.info("3. Ensure you're logged in: wandb login")
            log.info("4. Verify entity matches: entity should be 'mlops-group-85'")
            log.info("5. You can manually link artifacts in the WandB UI if needed")

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    train()
