import subprocess

from invoke import task


@task
def setup_environment(c, cuda="11.8", force_rebuild=False):
    """Set up the conda environment.

    Args:
    ----
        c: Invoke context
        cuda: CUDA version to install
        force_rebuild: Force recreation of environment if it exists

    """
    env_name = "ml-template"

    # Check if environment exists
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if env_name in result.stdout and not force_rebuild:
        print(f"Environment {env_name} already exists. Use --force-rebuild to recreate it.")
        return

    # Remove existing environment if force_rebuild
    if force_rebuild:
        print(f"Removing existing environment {env_name}...")
        subprocess.run(["conda", "env", "remove", "--name", env_name])

    print("Creating conda environment...")
    subprocess.run(["conda", "env", "create", "-f", "environment.yaml"])


@task
def setup_precommit(c):
    """Set up pre-commit hooks.

    Args:
    ----
        c: Invoke context

    """
    subprocess.run(["pre-commit", "install"])


@task
def clean(c):
    """Clean up temporary files.

    Args:
    ----
        c: Invoke context

    """
    patterns = ["*.pyc", "__pycache__", "*.egg-info", "dist", "build"]
    for pattern in patterns:
        c.run(f"find . -type f -name '{pattern}' -delete")
        c.run(f"find . -type d -name '{pattern}' -exec rm -rf {{}} +")


@task
def lint(c):
    """Run code quality tools.

    Args:
    ----
        c: Invoke context

    """
    print("Running black...")
    c.run("black .")
    print("Running ruff...")
    c.run("ruff check . --fix")
    print("Running mypy...")
    c.run("mypy src/")


@task
def test(c):
    """Run tests.

    Args:
    ----
        c: Invoke context

    """
    c.run("pytest tests/ -v")


@task
def generate_docs(c):
    """Generate project structure documentation.

    Args:
    ----
        c: Invoke context

    """
    print("Generating project structure documentation...")
    from src.utils.project_structure import ProjectStructureGenerator

    generator = ProjectStructureGenerator(".")
    generator.save_structure()
