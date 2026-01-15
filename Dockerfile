# Use a lightweight Python base image
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv

COPY README.md ./

COPY src ./src

RUN uv pip install --system --no-cache .

# Create models directory
RUN mkdir -p models

# Download model from WandB registry
# Build args for WandB configuration
ARG WANDB_API_KEY
ARG WANDB_ENTITY=mlops-group-85
ARG WANDB_PROJECT=mlops-project
ARG WANDB_ARTIFACT=best_model:latest

# Set environment variables for WandB
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV WANDB_ENTITY=${WANDB_ENTITY}
ENV WANDB_PROJECT=${WANDB_PROJECT}
ENV WANDB_ARTIFACT=${WANDB_ARTIFACT}

# Copy download script
COPY download_model.py ./

# Download model from WandB using Python script
# Note: wandb is already installed via uv pip install above
RUN python3 download_model.py

EXPOSE 8080

CMD ["uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
