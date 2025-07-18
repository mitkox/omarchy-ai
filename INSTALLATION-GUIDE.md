# Omarchy AI - Seamless Installation Guide

## Quick Installation (Recommended)

### Single Command Installation
```bash
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/refs/heads/master/boot.sh | bash
```

This command will:
- ✅ Download the latest Omarchy AI
- ✅ Install base Omarchy system (Hyprland, Waybar, etc.)
- ✅ Install AI development tools (Python, PyTorch, etc.)
- ✅ Create conda environment for AI development
- ✅ Setup Jupyter Lab and ML libraries
- ✅ Configure GPU support and monitoring
- ✅ Create AI workspace directory
- ✅ Setup shell aliases and commands

### Installation Time
- **Fresh system**: 20-40 minutes
- **Existing Arch system**: 10-20 minutes
- **Network dependent**: Package downloads

## What Gets Installed

### Base Omarchy System
- Hyprland (Wayland compositor)
- Waybar (status bar)
- Wofi (application launcher)
- Alacritty (terminal)
- Neovim (editor)
- Theme management system

### AI Development Environment
- **Python 3.11** in dedicated conda environment
- **PyTorch** with CUDA support
- **Jupyter Lab** for interactive development
- **Transformers** (Hugging Face)
- **MLflow** for experiment tracking
- **TensorBoard** for visualization
- **Various ML libraries** (numpy, pandas, scikit-learn, etc.)

### AI Tools & Commands
- `ai-env` - Activate AI development environment
- `ai-workspace` - Navigate to AI workspace
- `jupyter-ai` - Start Jupyter Lab
- `mlflow-ui` - Start MLflow UI
- `tensorboard-ai` - Start TensorBoard
- `gpu-monitor` - Monitor GPU usage

## Post-Installation Usage

### 1. Reboot (Required)
```bash
sudo reboot
```

### 2. Activate AI Environment
```bash
# Open a new terminal and run:
ai-env
```

### 3. Verify Installation
```bash
python --version    # Should show Python 3.11.x
pip list           # Shows installed packages
nvidia-smi         # Check GPU (if available)
```

### 4. Start Developing
```bash
ai-workspace       # Go to AI workspace directory
jupyter-ai         # Start Jupyter Lab
```

## AI Workspace Structure

After installation, you'll have:
```
~/ai-workspace/
├── projects/      # Your AI projects
├── models/        # Downloaded models
├── datasets/      # Training data
├── experiments/   # ML experiments
├── notebooks/     # Jupyter notebooks
├── logs/          # Training logs
├── mlruns/        # MLflow tracking
└── .env          # Environment variables
```

## Troubleshooting

### If `ai-env` doesn't work:
```bash
# Method 1: Manual activation
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai-dev

# Method 2: Fresh shell
bash --login
ai-env

# Method 3: Direct script
~/.local/share/omarchy-ai/activate-ai-env.sh
```

### If conda is not found:
```bash
# Ensure conda is in PATH
echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### If packages are missing:
```bash
ai-env
pip install transformers torch torchvision torchaudio
```

## Manual Installation (Advanced)

### 1. Clone Repository
```bash
git clone https://github.com/mitkox/omarchy-ai.git ~/.local/share/omarchy-ai
cd ~/.local/share/omarchy-ai
```

### 2. Run Installation
```bash
chmod +x install.sh
./install.sh
```

### 3. Follow post-installation steps above

## Key Features

### 🔒 **Privacy-First**
- Local AI inference with llama.cpp
- Offline-first development
- No cloud dependencies required

### 🚀 **Performance**
- CUDA GPU acceleration
- Optimized for AI workloads
- Real-time monitoring

### 🛠️ **Developer-Friendly**
- Pre-configured development environment
- Integrated tools and libraries
- Version control with Git LFS

### 📚 **Complete Ecosystem**
- Jupyter Lab for interactive development
- MLflow for experiment tracking
- TensorBoard for visualization
- Container support for deployment

## System Requirements

### Minimum
- **OS**: Arch Linux (fresh installation recommended)
- **RAM**: 8GB (16GB recommended for AI workloads)
- **Storage**: 50GB free space
- **Network**: Internet connection for installation

### Recommended
- **OS**: Fresh Arch Linux installation
- **RAM**: 32GB+ for large models
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: SSD with 100GB+ free space

## Support

### Getting Help
- Check the troubleshooting section above
- Review logs in `~/.local/share/omarchy-ai/logs/`
- Open an issue on GitHub

### Common Issues
- **Installation fails**: Check internet connection and disk space
- **GPU not detected**: Install NVIDIA drivers first
- **Conda issues**: Reboot and try again
- **Permission errors**: Ensure user has sudo access

## What's Next?

After successful installation:

1. **Download your first model**:
   ```bash
   ai-env
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen3-0.6B
   ```

2. **Start your first project**:
   ```bash
   ai-workspace
   mkdir my-first-ai-project
   cd my-first-ai-project
   jupyter-ai
   ```

3. **Explore the ecosystem**:
   - Try the pre-installed packages
   - Check out the example notebooks
   - Monitor your GPU usage
   - Set up experiment tracking

Welcome to Omarchy AI - your complete local AI development environment! 🚀
