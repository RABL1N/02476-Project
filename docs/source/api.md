# API and Frontend

## Public API and Frontend

The application is deployed to Google Cloud Run and is publicly accessible:

### API Endpoint

The FastAPI backend is available at:

- **Base URL:** `https://mlops-fastapi-304008424690.europe-west1.run.app`
- **Health Check:** `https://mlops-fastapi-304008424690.europe-west1.run.app/health`
- **Prediction Endpoint:** `https://mlops-fastapi-304008424690.europe-west1.run.app/predict`

### Frontend

The Streamlit web interface is available at:

- **Frontend URL:** `https://mlops-frontend-304008424690.europe-west1.run.app`

The frontend allows you to upload chest X-ray images and get predictions directly in your browser.

## Using the API

### Health Check

```bash
curl https://mlops-fastapi-304008424690.europe-west1.run.app/health
```

### Make a Prediction

```bash
curl -X POST "https://mlops-fastapi-304008424690.europe-west1.run.app/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpeg"
```

The API will return a JSON response:

```json
{
  "prediction": "NORMAL"
}
```

or

```json
{
  "prediction": "PNEUMONIA"
}
```

### Model Information

You can check which model is being used:

```bash
curl http://localhost:8080/model/info | python -m json.tool
```

This endpoint shows:

- Model path and size
- WandB artifact information (name, version, validation accuracy, loss)
- Model architecture details

Example response:

```json
{
  "wandb_artifact": {
    "name": "best_model:best",
    "version": "v0",
    "validation_accuracy": 81.25,
    "best_val_loss": 0.23879611492156985
  }
}
```

This confirms the container is using the model with the "best" alias from WandB, which has the highest validation accuracy among all trained models.
