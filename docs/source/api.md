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

### Data Drift Detection

The API provides an endpoint to detect data drift by comparing current feature distributions against a reference dataset. This helps monitor whether incoming data has shifted from the training distribution.

**Endpoint:** `POST /drift/features`

**Request Body:**

The endpoint expects a JSON payload with a list of feature rows. Each row should contain statistical features extracted from images:

```json
{
  "rows": [
    {
      "mean_intensity": 0.42,
      "std_intensity": 0.18,
      "min_intensity": 0.0,
      "max_intensity": 0.97
    },
    {
      "mean_intensity": 0.45,
      "std_intensity": 0.20,
      "min_intensity": 0.0,
      "max_intensity": 0.95
    }
  ]
}
```

**Example Request:**

```bash
curl -X POST "https://mlops-fastapi-304008424690.europe-west1.run.app/drift/features" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "mean_intensity": 0.42,
        "std_intensity": 0.18,
        "min_intensity": 0.0,
        "max_intensity": 0.97
      }
    ]
  }'
```

**Response:**

The API returns drift detection results using Evidently AI:

```json
{
  "dataset_drift": true,
  "number_of_drifted_columns": 4,
  "number_of_columns": 4,
  "share_of_drifted_columns": 1.0,
  "reference_rows": 5216,
  "current_rows": 1,
  "reference_path": "/app/src/mlops_project/monitoring/artifacts/reference_features.csv"
}
```

**Response Fields:**

- `dataset_drift`: Boolean indicating whether drift was detected
- `number_of_drifted_columns`: Number of columns that showed drift
- `number_of_columns`: Total number of columns analyzed
- `share_of_drifted_columns`: Proportion of columns that drifted (0.0 to 1.0)
- `reference_rows`: Number of rows in the reference dataset
- `current_rows`: Number of rows in the current request
- `reference_path`: Path to the reference features file used for comparison

**Note:** The reference dataset is generated from the training data and is stored in the container. To update the reference dataset, rebuild the Docker image after running the feature extraction task locally.

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
