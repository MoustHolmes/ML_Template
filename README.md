# ML Research Template

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A streamlined template for academic research and machine learning projects, built with PyTorch Lightning and Hydra.

## Features

- **Experiment Management**:
  - Hydra for configuration management
  - Weights & Biases for experiment tracking
  - Reproducible experiments with seed control
- **Code Quality**:
  - Type checking with MyPy
  - Linting with Ruff
  - Formatting with Black
  - Pre-commit hooks for code quality
- **Development Workflow**:
  - Conda environment management
  - Automated task running with Invoke
  - Comprehensive testing setup with PyTest
- **ML Framework**:
  - PyTorch Lightning for structured training
  - Hydra for hyperparameter management
  - Easy-to-extend modular design

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ml-template.git
cd ml-template
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate ml-template
```

3. Install development tools:
```bash
invoke setup-precommit
```

4. Run the example training:
```bash
python src/train.py
```

## Project Structure

```
├── configs/           # Hydra configuration files
│   ├── callbacks/     # Training callbacks
│   ├── data/         # Dataset configurations
│   ├── logger/       # Logging configurations
│   ├── model/        # Model architectures
│   └── trainer/      # Training strategies
├── src/              # Source code
│   ├── data/         # Data modules and processing
│   └── models/       # Lightning modules & components
└── tests/            # Test files
```

## Development Tools

Run common development tasks with Invoke:

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

## Configuration

The project uses Hydra for configuration management. Main config files:

- `configs/train.yaml`: Main training configuration
- `configs/data/`: Dataset-specific configs
- `configs/model/`: Model architecture configs
- `configs/trainer/`: Training strategy configs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
