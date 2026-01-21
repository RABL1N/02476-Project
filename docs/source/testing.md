# Testing

## Automated Testing of Staged Models

When you train a model using:

```bash
uv run invoke train
```

The trained model is automatically saved as an artifact on Weights & Biases (WandB). This triggers a webhook to GitHub, which starts a container that runs automated tests on the model. These tests are defined in [staged_model_tests/test_staged_model.py](staged_model_tests/test_staged_model.py).

This workflow ensures that every staged model is validated with a suite of checks before being promoted or used in production.

## Unit Tests

The project uses `pytest` for unit testing. Tests are located in the `tests/` directory and cover:

- Data loading and preprocessing (`test_data.py`)
- Model construction (`test_model.py`)
- Training procedures (`test_train.py`)
- API endpoints (`test_api.py`)

To run all tests:

```bash
uv run pytest tests/
```

To run tests with coverage:

```bash
uv run coverage run --source=src/mlops_project -m pytest tests/ -v
uv run coverage report -m
```

## Continuous Integration

The project uses GitHub Actions for continuous integration. Tests are automatically run on:

- Push to `main` branch
- Pull requests to `main` branch
- Multiple operating systems (Ubuntu, macOS)
- Multiple Python versions

Test results and coverage reports are automatically uploaded to Codecov.
