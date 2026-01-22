import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed",
        echo=True,
        pty=not WINDOWS,
    )


@task(help={"fake_training": "Enable fast/fake training mode for CI or testing."})
def train(ctx: Context, fake_training: bool = False) -> None:
    """Train model. Use --fake-training to enable fast/fake mode."""
    cmd = f"uv run src/{PROJECT_NAME}/train.py"
    if fake_training:
        cmd += " fake_training=true"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(help={"fake_training": "Enable fast/fake training mode for CI or testing."})
def lightning_train(ctx: Context, fake_training: bool = False) -> None:
    """Train model. Use --fake-training to enable fast/fake mode."""
    cmd = f"uv run src/{PROJECT_NAME}/train_lightning.py"
    if fake_training:
        cmd += " fake_training=true"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def download_model(ctx: Context, artifact: str = "best_model:best") -> None:
    """Download the best model from WandB for local inference."""
    import os
    from pathlib import Path

    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)

    # Set environment variables if not already set
    env_vars = {}
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set. Make sure you're logged in with 'wandb login'")

    ctx.run(
        "uv run python download_model.py",
        echo=True,
        pty=not WINDOWS,
        env={"WANDB_ARTIFACT": artifact, **env_vars},
    )


@task
def delete_artifacts(ctx: Context) -> None:
    """Delete all versions of best_model artifact from WandB (requires confirmation)."""
    ctx.run("uv run python delete_artifacts.py", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


# Drift detection commands
@task
def extract_drift_features(ctx: Context) -> None:
    """Extract features from datasets for drift detection."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.monitoring.extract_features",
        echo=True,
        pty=not WINDOWS,
    )


@task
def check_drift(ctx: Context) -> None:
    """Run drift analysis and generate HTML report."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.monitoring.drift_analysis",
        echo=True,
        pty=not WINDOWS,
    )


@task
def drift(ctx: Context) -> None:
    """Extract features and run drift analysis (full drift check)."""
    extract_drift_features(ctx)
    check_drift(ctx)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
