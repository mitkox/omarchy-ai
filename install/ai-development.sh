# AI Development Environment Setup for Omarchy AI
# This script installs core AI/ML tools and Python environment

echo "🤖 Setting up AI Development Environment..."

# Install Python and package management tools
echo "📦 Installing system packages..."
yay -S --noconfirm --needed \
  python python-pip python-pipx \
  python-virtualenv python-conda \
  miniconda3 \
  jupyter-notebook \
  python-numpy python-scipy python-matplotlib \
  python-pandas python-scikit-learn \
  python-seaborn python-plotly \
  python-requests python-urllib3 \
  python-yaml python-toml \
  python-rich python-tqdm \
  sqlite3 \
  curl wget \
  git-lfs \
  jq 2>/dev/null || echo "⚠️  Some system packages may have failed to install"

# Set environment variables for conda
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export CONDA_AUTO_ACTIVATE_BASE=false

# Initialize conda for this session
echo "🐍 Initializing conda..."
conda_found=false
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda_found=true
elif [ -f /usr/share/miniconda3/etc/profile.d/conda.sh ]; then
  source /usr/share/miniconda3/etc/profile.d/conda.sh
  conda_found=true
else
  echo "⚠️  Conda not found at expected locations. Checking PATH..."
  if command -v conda >/dev/null 2>&1; then
    # Try to initialize conda from wherever it is
    conda_base=$(conda info --base 2>/dev/null)
    if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
      source "$conda_base/etc/profile.d/conda.sh"
      conda_found=true
    fi
  fi
fi

if [ "$conda_found" = false ]; then
  echo "❌ Conda not available. AI packages will need to be installed manually."
  echo "💡 Try installing miniconda3 manually and re-running this script."
  return 1 2>/dev/null || exit 1
fi

# Create conda environment for AI development
echo "🔧 Creating AI development environment..."
if conda info --envs 2>/dev/null | grep -q "ai-dev"; then
  echo "ℹ️  AI development environment already exists, updating..."
else
  echo "📝 Creating new conda environment 'ai-dev' with Python 3.11..."
  conda create -n ai-dev python=3.11 -y -q 2>/dev/null || {
    echo "⚠️  Failed to create conda environment. Trying with verbose output..."
    conda create -n ai-dev python=3.11 -y
  }
fi

# Activate the environment for package installation
echo "🔄 Activating AI development environment..."
if conda activate ai-dev 2>/dev/null; then
  echo "✅ Environment activated successfully"
else
  echo "⚠️  Could not activate environment in current shell, packages will install to base"
fi

# Install essential packages in the conda environment
echo "📚 Installing essential AI/ML packages..."
conda install -c conda-forge -y -q \
  numpy scipy pandas matplotlib seaborn \
  scikit-learn jupyter ipython \
  pip setuptools wheel 2>/dev/null || {
  echo "⚠️  Some conda packages failed, installing with pip fallback..."
  pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyter ipython
}

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y -q 2>/dev/null || {
  echo "⚠️  Conda PyTorch install failed, trying pip..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "⚠️  PyTorch CUDA installation failed - install manually later with:"
    echo "      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
  }
}

# Install additional AI/ML libraries with pip
echo "🚀 Installing additional AI/ML libraries..."
ai_packages=(
  "transformers>=4.30.0"
  "datasets"
  "tokenizers"
  "accelerate" 
  "diffusers"
  "mlflow"
  "tensorboard"
  "langchain"
  "langchain-community"
  "sentence-transformers"
  "faiss-cpu"
  "gradio"
  "streamlit"
  "jupyterlab"
  "jupyterlab-git"
  "nbconvert"
  "nbformat"
  "ruff"
  "black"
  "flake8" 
  "isort"
  "mypy"
  "pytest"
  "pytest-cov"
  "pre-commit"
)

# Install packages in batches to handle failures gracefully
for package in "${ai_packages[@]}"; do
  pip install "$package" -q 2>/dev/null || echo "⚠️  Failed to install $package - install manually if needed"
done

# Deactivate environment after installation
conda deactivate 2>/dev/null || true

# Create AI workspace directory structure
echo "📁 Setting up AI workspace..."
mkdir -p ~/ai-workspace/{projects,models,datasets,experiments,notebooks,logs,mlruns}

# Create default .env file for AI projects
echo "📝 Creating AI environment configuration..."
cat > ~/ai-workspace/.env << 'EOF'
# AI Development Environment Variables
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
WANDB_MODE=offline
MLFLOW_TRACKING_URI=file:///home/${USER}/ai-workspace/mlruns
HF_HOME=/home/${USER}/ai-workspace/models/huggingface
TRANSFORMERS_CACHE=/home/${USER}/ai-workspace/models/transformers
JUPYTER_CONFIG_DIR=/home/${USER}/ai-workspace/.jupyter
EOF

# Create AI development aliases in bashrc (avoid duplicates)
echo "⚙️  Configuring AI development aliases..."
ai_aliases_marker="# AI Development Aliases - Omarchy AI"

# Remove old aliases if they exist
if grep -q "$ai_aliases_marker" ~/.bashrc; then
  echo "🔄 Updating existing AI aliases..."
  # Create temp file without the AI aliases section
  sed '/# AI Development Aliases - Omarchy AI/,/# End AI Development Aliases/d' ~/.bashrc > ~/.bashrc.tmp
  mv ~/.bashrc.tmp ~/.bashrc
fi

# Add new aliases
cat >> ~/.bashrc << EOF

$ai_aliases_marker
# Robust AI environment activation that handles various scenarios
ai-env() {
    # Set OpenSSL workaround
    export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    
    # Find and source conda
    conda_script=""
    if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
        conda_script="/opt/miniconda3/etc/profile.d/conda.sh"
    elif [ -f /usr/share/miniconda3/etc/profile.d/conda.sh ]; then
        conda_script="/usr/share/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        conda_base=\$(conda info --base 2>/dev/null)
        if [ -n "\$conda_base" ] && [ -f "\$conda_base/etc/profile.d/conda.sh" ]; then
            conda_script="\$conda_base/etc/profile.d/conda.sh"
        fi
    fi
    
    if [ -n "\$conda_script" ]; then
        source "\$conda_script"
        if conda activate ai-dev 2>/dev/null; then
            echo "✅ AI development environment activated!"
            echo "🐍 Python: \$(python --version 2>/dev/null || echo 'Not available')"
            echo "📦 Conda env: \$(conda info --envs | grep '*' | awk '{print \$1}' 2>/dev/null || echo 'Unknown')"
        else
            echo "❌ Failed to activate ai-dev environment"
            echo "💡 Try: conda create -n ai-dev python=3.11 -y"
        fi
    else
        echo "❌ Conda not found. Please install miniconda3 first."
    fi
}

# Other AI development aliases
alias ai-workspace='cd ~/ai-workspace'
alias jupyter-ai='cd ~/ai-workspace && jupyter lab'
alias mlflow-ui='mlflow ui --backend-store-uri file:///home/\${USER}/ai-workspace/mlruns'
alias tensorboard-ai='tensorboard --logdir ~/ai-workspace/logs'
# End AI Development Aliases
EOF

echo "✅ AI development environment setup complete!"
echo ""
echo "🎯 Usage:"
echo "  ai-env          - Activate AI development environment"
echo "  ai-workspace    - Go to AI workspace directory" 
echo "  jupyter-ai      - Start Jupyter Lab in AI workspace"
echo "  mlflow-ui       - Start MLflow UI"
echo "  tensorboard-ai  - Start TensorBoard"
echo ""
echo "📁 AI Workspace: ~/ai-workspace/"
echo "🐍 Environment: ai-dev (Python 3.11)"

# Create AI workspace directory
mkdir -p ~/ai-workspace/{projects,models,datasets,experiments,notebooks}

# Create default .env file for AI projects
cat > ~/ai-workspace/.env << 'EOF'
# AI Development Environment Variables
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
WANDB_MODE=offline
MLFLOW_TRACKING_URI=file:///home/${USER}/ai-workspace/mlruns
HF_HOME=/home/${USER}/ai-workspace/models/huggingface
TRANSFORMERS_CACHE=/home/${USER}/ai-workspace/models/transformers
EOF

# Create AI development aliases
cat >> ~/.bashrc << 'EOF'

# AI Development Aliases
alias ai-env='conda activate ai-dev'
alias ai-workspace='cd ~/ai-workspace'
alias jupyter-ai='cd ~/ai-workspace && jupyter lab'
alias mlflow-ui='mlflow ui --backend-store-uri file:///home/${USER}/ai-workspace/mlruns'
alias tensorboard-ai='tensorboard --logdir ~/ai-workspace/logs'
EOF

echo "AI development environment setup complete!"