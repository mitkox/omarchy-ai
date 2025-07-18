# AI CI/CD Pipeline Setup for Omarchy AI
# Creates local continuous integration and deployment tools for AI projects

# Install CI/CD tools
yay -S --noconfirm --needed \
  jenkins \
  drone \
  act-bin \
  mkcert \
  nginx

# Install Python testing and quality tools
pip install \
  pytest pytest-cov pytest-xdist pytest-mock \
  hypothesis \
  bandit safety \
  pre-commit \
  black isort flake8 mypy ruff \
  pytest-benchmark \
  pytest-html \
  coverage \
  tox

# Create AI project template with CI/CD
mkdir -p ~/ai-workspace/templates/ai-project

# Create project structure
cat > ~/ai-workspace/templates/ai-project/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-project"
version = "0.1.0"
description = "AI Project Template"
authors = [{name = "AI Developer", email = "developer@example.com"}]
dependencies = [
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "pandas",
    "scikit-learn",
    "mlflow",
    "click",
    "pydantic",
    "rich",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "isort",
    "flake8",
    "mypy",
    "pre-commit",
    "bandit",
    "safety",
]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --cov=src --cov-report=term-missing --cov-report=html"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "gpu: marks tests that require GPU",
]

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "setup.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
EOF

# Create GitHub Actions workflow
mkdir -p ~/ai-workspace/templates/ai-project/.github/workflows
cat > ~/ai-workspace/templates/ai-project/.github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Lint with flake8
      run: |
        flake8 src tests
    
    - name: Format check with black
      run: |
        black --check src tests
    
    - name: Import sort check
      run: |
        isort --check-only src tests
    
    - name: Type check with mypy
      run: |
        mypy src
    
    - name: Security check with bandit
      run: |
        bandit -r src
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  model-validation:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run model validation tests
      run: |
        pytest tests/test_model_validation.py -v --tb=short
    
    - name: Generate model performance report
      run: |
        python scripts/generate_model_report.py
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          models/
          reports/
EOF

# Create pre-commit configuration
cat > ~/ai-workspace/templates/ai-project/.pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first
      - id: requirements-txt-fixer

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]

  - repo: https://github.com/Lucas-C/pre-commit-hooks-safety
    rev: v1.3.2
    hooks:
      - id: python-safety-dependencies-check
EOF

# Create Makefile for common tasks
cat > ~/ai-workspace/templates/ai-project/Makefile << 'EOF'
.PHONY: help install test lint format type-check security clean build
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

test-gpu: ## Run GPU-specific tests
	pytest tests/ -v -m gpu

lint: ## Run linting
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format: ## Format code
	black src tests
	isort src tests

type-check: ## Run type checking
	mypy src

security: ## Run security checks
	bandit -r src
	safety check

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

model-train: ## Train model
	python scripts/train_model.py

model-evaluate: ## Evaluate model
	python scripts/evaluate_model.py

model-serve: ## Serve model
	python scripts/serve_model.py

docker-build: ## Build Docker image
	docker build -t ai-project .

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 ai-project
EOF

# Create local Jenkins pipeline
cat > ~/ai-workspace/templates/ai-project/Jenkinsfile << 'EOF'
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
        VENV_NAME = 'ai-project-env'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m venv ${VENV_NAME}
                    source ${VENV_NAME}/bin/activate
                    pip install --upgrade pip
                    pip install -e .[dev]
                '''
            }
        }
        
        stage('Lint') {
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make lint
                '''
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make test
                '''
            }
        }
        
        stage('Security') {
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make security
                '''
            }
        }
        
        stage('Model Training') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make model-train
                '''
            }
        }
        
        stage('Model Evaluation') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make model-evaluate
                '''
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    source ${VENV_NAME}/bin/activate
                    make build
                    make docker-build
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'htmlcov/**/*', allowEmptyArchive: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
        
        cleanup {
            sh 'rm -rf ${VENV_NAME}'
        }
    }
}
EOF

# Create project initialization script
cat > ~/ai-workspace/tools/init-ai-project.sh << 'EOF'
#!/bin/bash
# Initialize a new AI project with CI/CD setup

PROJECT_NAME="${1:-ai-project}"
PROJECT_DIR="$HOME/ai-workspace/projects/$PROJECT_NAME"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <project-name>"
    exit 1
fi

if [[ -d "$PROJECT_DIR" ]]; then
    echo "Project directory already exists: $PROJECT_DIR"
    exit 1
fi

echo "Creating AI project: $PROJECT_NAME"

# Copy template
cp -r ~/ai-workspace/templates/ai-project "$PROJECT_DIR"

# Initialize git repository
cd "$PROJECT_DIR"
git init
git add .
git commit -m "Initial commit: AI project template"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Create basic project structure
mkdir -p src/$PROJECT_NAME
mkdir -p tests
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p scripts
mkdir -p notebooks
mkdir -p reports

# Create __init__.py files
touch src/$PROJECT_NAME/__init__.py
touch tests/__init__.py

# Create basic module files
cat > src/$PROJECT_NAME/main.py << 'PYEOF'
"""Main module for the AI project."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option('--name', default='World', help='Name to greet.')
def hello(name):
    """Simple program that greets NAME."""
    console.print(f'Hello {name}!', style='bold green')


if __name__ == '__main__':
    hello()
PYEOF

# Create basic test file
cat > tests/test_main.py << 'PYEOF'
"""Tests for main module."""

import pytest
from click.testing import CliRunner

from src.ai_project.main import hello


def test_hello_default():
    """Test hello command with default name."""
    runner = CliRunner()
    result = runner.invoke(hello)
    assert result.exit_code == 0
    assert 'Hello World!' in result.output


def test_hello_custom_name():
    """Test hello command with custom name."""
    runner = CliRunner()
    result = runner.invoke(hello, ['--name', 'Alice'])
    assert result.exit_code == 0
    assert 'Hello Alice!' in result.output
PYEOF

# Create README
cat > README.md << 'MDEOF'
# AI Project

A machine learning project built with Omarchy AI.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .[dev]
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development

### Running Tests
```bash
make test
```

### Linting and Formatting
```bash
make lint
make format
```

### Type Checking
```bash
make type-check
```

### Security Checks
```bash
make security
```

## Model Development

### Training
```bash
make model-train
```

### Evaluation
```bash
make model-evaluate
```

### Serving
```bash
make model-serve
```

## CI/CD

This project includes:
- GitHub Actions workflows for CI/CD
- Pre-commit hooks for code quality
- Local Jenkins pipeline support
- Docker containerization
- Model validation and testing

## Project Structure
```
├── src/               # Source code
├── tests/             # Test files
├── data/              # Data files
├── models/            # Trained models
├── scripts/           # Utility scripts
├── notebooks/         # Jupyter notebooks
├── reports/           # Generated reports
└── .github/           # GitHub Actions workflows
```
MDEOF

echo "Project $PROJECT_NAME created successfully!"
echo "Location: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. source venv/bin/activate"
echo "3. make test"
echo "4. Start developing your AI project!"
EOF

chmod +x ~/ai-workspace/tools/init-ai-project.sh

# Add CI/CD aliases
cat >> ~/.bashrc << 'EOF'

# AI CI/CD Aliases
alias ai-init='~/ai-workspace/tools/init-ai-project.sh'
alias ai-test='cd ~/ai-workspace && make test'
alias ai-lint='cd ~/ai-workspace && make lint'
alias ai-format='cd ~/ai-workspace && make format'
alias ai-security='cd ~/ai-workspace && make security'
EOF

echo "AI CI/CD pipeline setup complete!"
echo "Available commands:"
echo "  ai-init <project-name> - Initialize new AI project"
echo "  ai-test                - Run tests"
echo "  ai-lint                - Run linting"
echo "  ai-format              - Format code"
echo "  ai-security            - Run security checks"