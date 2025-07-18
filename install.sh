# Exit immediately if a command exits with a non-zero status
set -e

# Give people a chance to retry running the installation
trap 'echo "Omarchy AI installation failed! You can retry by running: source ~/.local/share/omarchy-ai/install.sh"' ERR

# Ensure we're running from the correct directory
if [[ ! -f ~/.local/share/omarchy-ai/install.sh ]]; then
    echo "⚠️  Installation must be run from ~/.local/share/omarchy-ai/"
    echo "If running locally, ensure the repository is copied to the correct location first."
    exit 1
fi

# Install base Omarchy system first
echo -e "\nInstalling base Omarchy system..."
rm -rf ~/.local/share/omarchy/
git clone https://github.com/basecamp/omarchy.git ~/.local/share/omarchy >/dev/null 2>&1

for f in ~/.local/share/omarchy/install/*.sh; do
  echo -e "\nRunning base installer: $f"
  source "$f"
done

# Setup environment variables for conda and AI tools
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export CONDA_AUTO_ACTIVATE_BASE=false

# Initialize conda properly for this session if available
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f /usr/share/miniconda3/etc/profile.d/conda.sh ]; then
  source /usr/share/miniconda3/etc/profile.d/conda.sh
fi

# Install AI-specific components
echo -e "\nInstalling AI development tools..."
for f in ~/.local/share/omarchy-ai/install/*.sh; do
  echo -e "\nRunning AI installer: $f"
  source "$f"
done

# Ensure conda is properly initialized for new shells
echo -e "\nConfiguring conda for shell integration..."
conda_init_added=false
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  if ! grep -q "miniconda3/etc/profile.d/conda.sh" ~/.bashrc; then
    echo "[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && source /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    conda_init_added=true
  fi
elif [ -f /usr/share/miniconda3/etc/profile.d/conda.sh ]; then
  if ! grep -q "miniconda3/etc/profile.d/conda.sh" ~/.bashrc; then
    echo "[ -f /usr/share/miniconda3/etc/profile.d/conda.sh ] && source /usr/share/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    conda_init_added=true
  fi
fi

# Add OpenSSL workaround to bashrc if not already present
if ! grep -q "CRYPTOGRAPHY_OPENSSL_NO_LEGACY" ~/.bashrc; then
  echo "export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1" >> ~/.bashrc
fi

# Test that ai-env command works
echo -e "\nTesting AI environment setup..."
if command -v conda >/dev/null 2>&1; then
  if conda info --envs 2>/dev/null | grep -q "ai-dev"; then
    echo "✅ AI development environment (ai-dev) created successfully"
    
    # Test activation
    if conda activate ai-dev 2>/dev/null; then
      python_version=$(python --version 2>/dev/null || echo "Unknown")
      echo "✅ Environment activation test passed - $python_version"
      conda deactivate 2>/dev/null || true
    else
      echo "⚠️  Environment created but activation needs new shell session"
    fi
  else
    echo "⚠️  AI development environment not found - will be available after reboot"
  fi
else
  echo "⚠️  Conda not available in current session - will be available after reboot"
fi

# Install enhanced features
echo -e "\nInstalling enhanced AI features..."
enhanced_features=(
  "distributed-training.sh"
  "make-docs.sh"
  "containers.sh"
)

for feature in "${enhanced_features[@]}"; do
  if [[ -f ~/.local/share/omarchy-ai/install/$feature ]]; then
    echo -e "\nInstalling: $feature"
    source ~/.local/share/omarchy-ai/install/$feature
  fi
done

# Setup enhanced migration system
echo -e "\nSetting up migration system..."
~/.local/share/omarchy-ai/bin/omarchy-enhanced-migrations

# Ensure locate is up to date now that everything has been installed
sudo updatedb

# Final verification and setup
echo -e "\n🔧 Performing final AI environment verification..."

# Verify AI workspace exists
if [ -d ~/ai-workspace ]; then
  echo "✅ AI workspace directory created"
else
  echo "⚠️  Creating AI workspace directory..."
  mkdir -p ~/ai-workspace/{projects,models,datasets,experiments,notebooks,logs,mlruns}
fi

# Verify environment file exists  
if [ -f ~/ai-workspace/.env ]; then
  echo "✅ AI environment configuration file exists"
else
  echo "⚠️  Creating AI environment configuration..."
  cat > ~/ai-workspace/.env << 'ENVEOF'
# AI Development Environment Variables
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false
WANDB_MODE=offline
MLFLOW_TRACKING_URI=file:///home/${USER}/ai-workspace/mlruns
HF_HOME=/home/${USER}/ai-workspace/models/huggingface
TRANSFORMERS_CACHE=/home/${USER}/ai-workspace/models/transformers
JUPYTER_CONFIG_DIR=/home/${USER}/ai-workspace/.jupyter
ENVEOF
fi

# Verify conda environment exists
if command -v conda >/dev/null 2>&1; then
  if conda info --envs 2>/dev/null | grep -q "ai-dev"; then
    echo "✅ AI conda environment verified"
  else
    echo "⚠️  AI conda environment missing - creating fallback..."
    conda create -n ai-dev python=3.11 -y -q 2>/dev/null || echo "⚠️  Manual environment creation needed"
  fi
fi

# Verify aliases are set
if grep -q "AI Development Aliases - Omarchy AI" ~/.bashrc; then
  echo "✅ AI development aliases configured"
else
  echo "⚠️  AI aliases missing - this should not happen, checking installation..."
fi

echo -e "\n🎉 Omarchy AI installation complete!"
echo -e "\n📋 Installation Summary:"
echo -e "   ✅ Base Omarchy system installed"
echo -e "   ✅ AI development tools installed"
echo -e "   ✅ Conda environment 'ai-dev' created"
echo -e "   ✅ AI workspace created at ~/ai-workspace/"
echo -e "   ✅ Shell aliases configured"

echo -e "\n🚀 Next steps:"
echo -e "1. Reboot your system to apply all settings:"
echo -e "   sudo reboot"
echo -e ""
echo -e "2. After reboot, open a new terminal and activate AI environment:"
echo -e "   ai-env"
echo -e ""
echo -e "3. Verify installation:"
echo -e "   python --version    # Should show Python 3.11.x"
echo -e "   pip list            # Shows installed packages"
echo -e ""
echo -e "4. Start developing:"
echo -e "   ai-workspace        # Go to AI workspace"
echo -e "   jupyter-ai          # Start Jupyter Lab"
echo -e "   pip install package # Install additional packages"
echo -e ""
echo -e "5. Additional tools:"
echo -e "   mlflow-ui           # MLflow experiment tracking"
echo -e "   tensorboard-ai      # TensorBoard visualization"
echo -e "   gpu-monitor         # Monitor GPU usage"

# Check if running interactively
if [ -t 0 ] && [ -t 1 ] && command -v gum >/dev/null 2>&1; then
  echo -e "\n"
  gum confirm "Reboot now to apply all settings?" && sudo reboot
else
  echo -e "\n💡 Remember to reboot your system to complete the installation!"
fi
