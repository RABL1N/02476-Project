# Deployment

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

**2. Make a prediction:**

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
   gcloud compute ssh --zone "europe-west1-d" "instance-20260113-110032" --project "machineoperationproject"
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
