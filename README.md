# ML Research Template

A streamlined template for academic research and Kaggle competitions, built with PyTorch Lightning and Hydra.

## Quick Setup

1. Create the conda environment:
```bash
conda env create -f environment.yaml
conda activate ml-template
```

2. Install development dependencies:
```bash
invoke setup-precommit
```

## Development Tools

- **Code Formatting**: Black (line-length: 99)
- **Linting**: Ruff for fast Python linting
- **Type Checking**: MyPy for static type checking
- **Testing**: PyTest
- **Pre-commit**: Automated code quality checks on commit
- **Task Runner**: Invoke for common development tasks

## Common Tasks

```bash
# Setup development environment
invoke setup-environment

# Run code quality tools
invoke lint

# Run tests
invoke test

# Clean up temporary files
invoke clean
```

## Project Structure

```
├── configs/           # Hydra configuration files
│   ├── callbacks/    # Training callbacks (early stopping, etc.)
│   ├── data/        # Dataset configurations
│   ├── logger/      # WandB and other logger configs
│   ├── model/       # Model architectures
│   └── trainer/     # Training strategies
├── src/             # Source code
│   ├── data/        # Dataset and DataModule definitions
│   └── models/      # Lightning modules and model components
└── tests/           # Test files
```

## Features

- **Experiment Management**: Hydra for configuration, WandB for logging
- **Reproducibility**: Conda environment and consistent code formatting
- **Development Workflow**: Pre-commit hooks and development tools
- **Type Safety**: Static type checking with MyPy
- **Testing**: PyTest integration
- **Task Automation**: Invoke for common development tasks

## Getting Started

[Installation and usage instructions will be added]

## Development Roadmap

1. Core Template Development
   - [ ] Basic project structure
   - [ ] Hydra configuration setup
   - [ ] PyTorch Lightning integration

2. MLOps Integration
   - [ ] WandB setup
   - [ ] Docker configuration
   - [ ] CI/CD with GitHub Actions

3. Example Implementation
   - [ ] Basic classification example
   - [ ] Advanced use cases

## Implementation Plan

### 1. Project Setup (Combining Best Practices)

1. **Project Structure**
   ```
   ├── configs/                    # Hydra configuration files
   │   ├── callbacks/
   │   ├── data/
   │   ├── experiment/
   │   ├── hparams_search/
   │   ├── model/
   │   └── trainer/
   ├── data/                      # Dataset storage
   │   ├── processed/
   │   └── raw/
   ├── src/                       # Source code
   │   ├── callbacks/             # Lightning callbacks
   │   ├── data/                  # DataModules
   │   ├── models/               # Lightning Modules
   │   ├── utils/                # Utility functions
   │   ├── train.py             # Training pipeline
   │   └── eval.py              # Evaluation pipeline
   ├── tests/                    # Testing suite
   ├── notebooks/                # Jupyter notebooks
   └── tasks.py                  # Invoke task definitions
   ```

2. **Core Components Integration**
   - Using Cookiecutter setup from mlops_template for project scaffolding
   - Implementing Hydra configuration system from Lightning-Hydra-Template
   - Integrating WandB logging with advanced features
   - Setting up Invoke tasks for common operations

### 2. Training Pipeline

The training pipeline follows a clean, modular approach:

```python
@hydra.main(config_path="../configs", config_name="train")
def train(config: DictConfig):
    # Set up logging and experiment tracking
    wandb_logger = WandbLogger(...)

    # Initialize data module
    datamodule = hydra.utils.instantiate(config.data)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # Initialize callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Initialize trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **config.trainer
    )

    # Train model
    trainer.fit(model, datamodule)
```

### 3. Key Features from Each Template

1. **From Lightning-Hydra-Template:**
   - Hydra configuration system
   - Lightning training structure
   - Experiment organization

2. **From MLOps Template:**
   - Project structure using Cookiecutter
   - Code quality tools (Ruff, pre-commit)
   - Docker setup

3. **From Starfish Detection:**
   - Training pipeline implementation
   - WandB integration
   - Custom callbacks

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML_Template
cd ML_Template

# Create conda environment
conda create -n ml_template python=3.10
conda activate ml_template

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Create New Project:**
```bash
cookiecutter https://github.com/yourusername/ML_Template
```

2. **Training:**
```bash
# Basic training
python src/train.py

# Train with specific experiment config
python src/train.py experiment=experiment_name

# Train with hyperparameter sweep
python src/train.py -m hparams_search=mnist_optuna
```

3. **Common Tasks with Invoke:**
```bash
# List all available tasks
invoke --list

# Run tests
invoke test

# Run code quality checks
invoke lint

# Clean project
invoke clean
```

## Development Workflow

1. **Setup Project:**
   ```bash
   # Initialize new project
   cookiecutter ML_Template
   cd your_project_name

   # Install pre-commit hooks
   pre-commit install
   ```

2. **Implement Components:**
   ```python
   # 1. Define DataModule in src/data/
   class MyDataModule(LightningDataModule):
       ...

   # 2. Define Model in src/models/
   class MyModel(LightningModule):
       ...

   # 3. Create configs in configs/
   # data/my_data.yaml
   # model/my_model.yaml
   ```

3. **Create Experiment:**
   ```yaml
   # configs/experiment/my_experiment.yaml
   defaults:
     - override /data: my_data
     - override /model: my_model
     - override /callbacks: default
     - override /trainer: default

   tags: ["experiment_name"]
   ```

4. **Run Training:**
   ```bash
   python src/train.py experiment=my_experiment
   ```

## Example Projects

Check the `examples/` directory for complete implementations:
- Basic MNIST Classification
- Time Series Forecasting
- Computer Vision Object Detection

Each example demonstrates different aspects of the template and can serve as a starting point for your own projects.
