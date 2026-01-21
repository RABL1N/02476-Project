# mlops_project

## Overall goal of the project

The goal of this project is to develop a machine learning model that can classify chest X ray images as either normal or showing signs of pneumonia. The main focus is on creating a reproducible and well structured machine learning workflow rather than achieving maximum performance.

## Data description

The dataset used in this project consists of chest X ray images from pediatric patients. The images are labeled as either normal or pneumonia. The dataset contains several thousand samples and is divided into training validation and test sets. Each sample is a grayscale image stored as an image file. The total dataset size is on the order of a few hundred megabytes. The data modality is image based and the task is binary classification.

## Expected models

The project will initially use a simple convolutional neural network for image classification. If time allows transfer learning with pretrained image classification models may be explored. The emphasis is on model training evaluation and reproducibility rather than model complexity.

## Quick start

This project exposes a simple command-line interface for running the main workflows.
All commands are executed from the project root using `uv`.

### Download and prepare the dataset

```bash
uv run python main.py data
```

### Train the model

```bash
uv run python main.py train
```

### Run unit tests

```bash
uv run python main.py test
```

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
