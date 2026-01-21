FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Backend dependencies
COPY backend/backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy backend code
COPY src ./src
COPY download_model.py .
COPY README.md .

RUN mkdir -p models

ARG WANDB_API_KEY
ARG WANDB_ENTITY=mlops-group-85
ARG WANDB_PROJECT=mlops-project
ARG WANDB_ARTIFACT=best_model:best

ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV WANDB_ENTITY=${WANDB_ENTITY}
ENV WANDB_PROJECT=${WANDB_PROJECT}
ENV WANDB_ARTIFACT=${WANDB_ARTIFACT}
ENV PYTHONPATH=/app/src

RUN python download_model.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.mlops_project.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
