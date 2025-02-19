# Project Structure

## Directory Structure and Code Analysis

📁 Project Root
  📁 configs/
    📁 callbacks/
      📄 default.yaml
    📁 data/
      📄 mnist.yaml
    📁 logger/
      📄 wandb.yaml
    📁 model/
      📄 mnist.yaml
    📁 paths/
      📄 default.yaml
    📁 trainer/
      📄 default.yaml
    📄 train.yaml
  📁 src/
    📁 data/
      📄 mnist_datamodule.py
          Classes:
           • MNISTDataModule
            No description

          Functions:
           • prepare_data
            No description

           • setup
            No description

           • train_dataloader
            No description

           • val_dataloader
            No description

           • test_dataloader
            No description

           • teardown
            Clean up after fit or test.

    📁 models/
      📁 components/
        📄 simple_dense_net.py
            Classes:
             • SimpleDenseNet
              A simple feedforward neural network with configurable layer sizes.

              This network consists of three hidden layers with ReLU activation functions,
              designed for classification tasks.

            Functions:
             • forward
              Forward pass through the network.

              Args:
              ----
              x: Input tensor of shape (batch_size, channels, height, width)

              Returns:
              -------
              Output tensor of shape (batch_size, output_size)

      📄 mnist_module.py
          Classes:
           • MNISTLitModule
            Implement a LightningModule for MNIST classification.

            Handles training, validation, and testing for MNIST digit classification.
            Implements basic training logic and metric tracking.

          Functions:
           • forward
            Perform forward pass through the network.

            Args:
            ----
            x: Input tensor of shape (B, 1, 28, 28)

            Returns:
            -------
            Logits tensor of shape (B, 10)

           • on_train_start
            Reset best validation accuracy at the start of training.

           • model_step
            Perform a single model step on a batch of data.

            Args:
            ----
            batch: Tuple of (images, labels)

            Returns:
            -------
            Tuple of (loss, predictions, targets)

           • training_step
            Execute training step for a single batch.

            Args:
            ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

            Returns:
            -------
            Computed loss value

           • validation_step
            Execute validation step for a single batch.

            Args:
            ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

           • on_validation_epoch_end
            Log best validation accuracy at the end of the validation epoch.

           • test_step
            Execute test step for a single batch.

            Args:
            ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

           • configure_optimizers
            Configure optimizers for training.

            Returns
            -------
            Dictionary containing the configured optimizer

    📁 utils/
      📄 project_structure.py
          Classes:
           • ProjectStructureGenerator
            No description

          Functions:
           • should_ignore
            Check if the path should be ignored based on gitignore and base patterns.

           • parse_python_file
            Parse a Python file and extract classes and functions with their docstrings.

           • generate_structure
            Generate the project structure with code analysis.

           • save_structure
            Generate and save the project structure to a markdown file.

           • add_to_output
            Recursively add directory contents to output with proper indentation.

    📄 train.py
        Functions:
         • train
          Execute the main training routine.

          Args:
          ----
          cfg: Configuration composed by Hydra.

          Returns:
          -------
          Optional float: Metric score for hyperparameter optimization.

  📁 templates/
    📁 lightning-hydra-template/
      📁 src/
  📄 environment.yaml
  📄 PROJECT_STRUCTURE.md
  📄 pyproject.toml
  📄 README.md
  📄 requirements.txt
  📄 tasks.py
      Functions:
       • setup_environment
        Set up the conda environment.

        Args:
        ----
        c: Invoke context
        cuda: CUDA version to install
        force_rebuild: Force recreation of environment if it exists

       • setup_precommit
        Set up pre-commit hooks.

        Args:
        ----
        c: Invoke context

       • clean
        Clean up temporary files.

        Args:
        ----
        c: Invoke context

       • lint
        Run code quality tools.

        Args:
        ----
        c: Invoke context

       • test
        Run tests.

        Args:
        ----
        c: Invoke context

       • generate_docs
        Generate project structure documentation.
        Args:
        ----
        c: Invoke context
