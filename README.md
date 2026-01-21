# mlops_project

### Overall goal of the project

The goal of this project is to develop a machine learning model that can classify chest X ray images as either normal or showing signs of pneumonia. The main focus is on creating a reproducible and well structured machine learning workflow rather than achieving maximum performance.

### Data description

The dataset used in this project consists of chest X ray images from pediatric patients. The images are labeled as either normal or pneumonia. The dataset contains several thousand samples and is divided into training validation and test sets. Each sample is a grayscale image stored as an image file. The total dataset size is on the order of a few hundred megabytes. The data modality is image based and the task is binary classification.

### Expected models

The project will initially use a simple convolutional neural network for image classification. If time allows transfer learning with pretrained image classification models may be explored. The emphasis is on model training evaluation and reproducibility rather than model complexity.

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project. The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
  are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [X] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [X] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Project structure

The directory structure of the project looks like this:

```txt
├── .devcontainer/              # Dev container configuration
│   ├── devcontainer.json
│   └── post_create.sh
├── .dvc/                       # DVC configuration
│   ├── .gitignore
│   └── config
├── .github/                    # Github actions and dependabot
│   ├── agents/
│   │   └── dtu_mlops_agent.md
│   ├── dependabot.yaml
│   ├── prompts/
│   │   └── add_test.prompt.md
│   └── workflows/
│       ├── data_changes.yaml
│       ├── linting.yaml
│       ├── stage_model.yaml
│       ├── test_staged_model.yaml
│       └── tests.yaml
├── backend/                    # Backend deployment files
│   ├── backend_requirements.txt
│   └── Dockerfile
├── configs/                    # Configuration files
│   ├── .gitkeep
│   └── config.yaml
├── data/                       # Data directory (version controlled with DVC)
│   └── raw.dvc
├── dockerfiles/                # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                       # Documentation
│   ├── mkdocs.yaml
│   ├── README.md
│   └── source/
│       └── index.md
├── frontend/                   # Frontend application
│   ├── Dockerfile
│   └── streamlit_app.py
├── load_tests/                 # Load testing with Locust
│   ├── locustfile.py
│   ├── test_image.png
│   └── reports/
├── models/                     # Trained models (not in git)
├── notebooks/                  # Jupyter notebooks
├── reports/                    # Reports and figures
│   ├── figures/
│   └── .gitkeep
├── src/                        # Source code
│   └── mlops_project/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── staged_model_tests/         # Tests for staged models
│   └── test_staged_model.py
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── AGENTS.md                   # Guidance for coding agents
├── dataloader.py
├── download_model.py
├── LICENSE
├── main.py                     # CLI entry point
├── pyproject.toml              # Python project configuration
├── README.md                   # This file
├── setup_vm_git.sh            # VM setup script
├── sync_vm.sh                 # VM sync script
├── tasks.py                   # Invoke task definitions
└── uv.lock                     # UV lock file
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).

## Application Usage

This project exposes a simple command-line interface for running the main workflows.
All commands are executed from the project root using `uv`.

### Download and prepare the dataset

```bash
uv run python main.py data
Train the model
uv run python main.py train
Run unit tests
uv run python main.py test
```

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

### Using the API

**Health Check:**

```bash
curl https://mlops-fastapi-304008424690.europe-west1.run.app/health
```

**Make a Prediction:**

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

## GCP VM Setup and Syncing

This project includes scripts to set up and sync the project with a Google Cloud Platform (GCP) VM instance for training.

### Initial Setup

1. **Set up the repository on the VM:**

   ```bash
   ./setup_vm_git.sh
   ```

   This script will:

   - Install git on the VM (if needed)
   - Clone the repository from GitHub to `~/mlops_project` on the VM
   - Install `uv` package manager
   - Configure PATH for `uv`
2. **Authenticate with Google Cloud Storage:**
   SSH into the VM and authenticate with your user credentials to access the DVC remote:

   ```bash
   gcloud compute ssh instance-20260113-110032 --zone=europe-west1-d
   gcloud auth application-default login
   ```

   This is a one-time setup. Credentials will persist across sessions.
3. **Install dependencies on the VM:**

   ```bash
   cd ~/mlops_project
   export PATH="$HOME/.local/bin:$PATH"  # If not already in PATH
   uv sync
   ```

### Syncing Changes

After making changes locally and pushing to GitHub, sync the VM with:

```bash
./sync_vm.sh
```

This script will:

- Pull the latest code changes from GitHub
- Pull the latest data files using DVC from Google Cloud Storage

**Note:** The sync script uses user credentials (Option 2) for GCS access. If you encounter authentication errors, re-authenticate on the VM with `gcloud auth application-default login`.

### Scripts Overview

- `setup_vm_git.sh` - Initial setup: clones repository, installs dependencies
- `sync_vm.sh` - Regular syncing: pulls latest code and data from GitHub and GCS

Both scripts are designed to be run from your **local terminal** and will remotely execute commands on the VM via SSH.

## Docker Deployment

The project includes a Docker setup for deploying the inference API. The Docker image automatically downloads the best model from WandB during the build process.

### How It Works

The Dockerfile (`Dockerfile` in the project root) is configured to:

1. Install all project dependencies
2. Download the model with the "best" alias from WandB (`best_model:best`)
3. Save the model to `models/best_model.pt` inside the container
4. Start the FastAPI inference server on port 8080

The model is downloaded from WandB artifacts during the Docker build, ensuring the container always uses the best-performing model based on validation accuracy.

### Building the Docker Image

To build the Docker image, you need to provide your WandB API key:

```bash
docker build -t pneumonia-api:latest \
  --build-arg WANDB_API_KEY=your_wandb_api_key_here \
  -f Dockerfile .
```

To get your WandB API key:

- Visit: https://wandb.ai/authorize
- Or check your WandB settings

**Note:** The Docker image tag (`:latest`, `:best`, etc.) is just a label and doesn't affect which model is downloaded. The Dockerfile always downloads `best_model:best` from WandB regardless of the tag name.

### Running the Container

Start the container:

```bash
docker run -d -p 8080:8080 --name pneumonia-api pneumonia-api:latest
```

- `-d`: Run in detached mode (background)
- `-p 8080:8080`: Map port 8080 from container to host
- `--name pneumonia-api`: Name for the container

### Checking Container Status

```bash
# Check if container is running
docker ps

# View container logs
docker logs pneumonia-api

# Stop the container
docker stop pneumonia-api

# Remove the container (after stopping)
docker rm pneumonia-api
```

### Running Inference

Once the container is running, you can make predictions:

**1. Check API health:**

```bash
curl http://localhost:8080/health
```

**2. Check which model is being used:**

```bash
curl http://localhost:8080/model/info | python -m json.tool
```

This endpoint shows:

- Model path and size
- WandB artifact information (name, version, validation accuracy, loss)
- Model architecture details

**3. Make a prediction:**

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpeg"
```

Example with a test image:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/chest_xray/test/NORMAL/IM-0031-0001.jpeg"
```

The response will be:

```json
{
  "prediction": "NORMAL",
  "class_index": 0
}
```

### Verifying the Best Model

The `/model/info` endpoint confirms which model is loaded. Look for the `wandb_artifact` section in the response:

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

### Automated testing of staged models

When you train a model using:

```bash
uv run invoke train
```

The trained model is automatically saved as an artifact on Weights & Biases (WandB). This triggers a webhook to GitHub, which starts a container that runs automated tests on the model. These tests are defined in [staged_model_tests/test_staged_model.py](staged_model_tests/test_staged_model.py).

This workflow ensures that every staged model is validated with a suite of checks before being promoted or used in production.
