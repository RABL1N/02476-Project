# Getting Started

## Application Usage

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

## Development Setup

The project uses `uv` for management of virtual environments. This means:

- To install packages, use `uv add <package-name>`.
- To run Python scripts, use `uv run <script-name>.py`.
- To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.

### Testing

The project uses `pytest` for testing. To run tests, use:

```bash
uv run pytest tests/
```

### Linting and Formatting

The project uses `ruff` for linting and formatting:

- To format code, use `uv run ruff format .`
- To lint code, use `uv run ruff check . --fix`

### Task Management

The project uses `invoke` for task management. To see available tasks, use:

```bash
uv run invoke --list
```

Or refer to the `tasks.py` file.

### Pre-commit Hooks

The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use:

```bash
uv run pre-commit run --all-files
```

For more information, refer to the `.pre-commit-config.yaml` file.
