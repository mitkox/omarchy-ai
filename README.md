# Omarchy AI

Turn a fresh Arch installation into a fully-configured, beautiful, and modern AI development system based on Hyprland by running a single command. Built on the foundation of Omarchy, Omarchy AI is specifically designed for AI engineers and researchers who need a complete, local-first development environment.

[![GitHub stars](https://img.shields.io/github/stars/mitkox/omarchy-ai?style=flat-square)](https://github.com/mitkox/omarchy-ai/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/mitkox/omarchy-ai?style=flat-square)](https://github.com/mitkox/omarchy-ai/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

Unlike cloud-based AI development, Omarchy AI provides:
- **Local AI inference** with llama.cpp and CUDA GPU support
- **Offline-first development** with cached models and documentation
- **Integrated CI/CD** for AI model testing and deployment
- **Model management** for downloading, versioning, and serving AI models
- **Privacy-focused** development with no cloud dependencies
- **Comprehensive diagnostics** and automatic repair tools
- **Complete testing framework** for validation and troubleshooting

Read more about the original Omarchy at [omarchy.org](https://omarchy.org).

## 🚀 Quick Start

### One-Command Installation
```bash
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash
```

**That's it!** In 15-30 minutes you'll have a complete AI development environment.

### 📋 Prerequisites
- Fresh Arch Linux installation
- 16GB RAM (32GB+ recommended)
- 100GB free disk space (500GB+ recommended)
- Internet connection for initial setup
- NVIDIA GPU (optional but recommended)

### ✅ Post-Installation
```bash
# After reboot, verify your installation
omarchy-ai-doctor

# Start developing
ai-env
jupyter-ai
```

📚 **New to Omarchy AI?** Check out the [Quick Start Guide](QUICKSTART.md) for a step-by-step tutorial.

## ✨ What's New in v2.0

### 🛠️ Enhanced Installation & Validation
- **System Requirements Validation**: Comprehensive pre-installation checks
- **Installation Recovery**: Rollback capability and checkpoint system
- **Smart Error Handling**: Detailed error messages with solutions
- **Progress Tracking**: Real-time installation progress indicators

### 🔍 Diagnostic & Repair Tools
- **omarchy-ai-doctor**: Complete system health diagnostics
- **omarchy-ai-repair**: Automatic issue detection and repair
- **omarchy-ai-test**: Comprehensive testing framework
- **Performance Benchmarking**: CPU and GPU performance validation

### 📦 Improved Dependency Management
- **Centralized Requirements**: All dependencies in `requirements.txt` and `environment.yml`
- **Version Pinning**: Stable, tested package versions
- **Smart Installation**: Handles package conflicts automatically
- **Cache Management**: Intelligent cleanup and optimization

### 📚 Better Documentation
- **Quick Start Guide**: Get running in 15 minutes
- **Troubleshooting Guide**: Solutions for common issues
- **API Documentation**: Complete command reference
- **Example Projects**: Ready-to-run AI demos

## 🎯 Features

### 🤖 AI Development Environment
- **Python ecosystem** with PyTorch, TensorFlow, and Hugging Face Transformers
- **Local model inference** with llama.cpp and CUDA GPU acceleration
- **Jupyter notebooks** for interactive AI development
- **Model management** system with Git LFS integration and version control
- **Offline documentation** for all major AI frameworks
- **Enhanced GPU monitoring** with real-time performance tracking
- **Distributed training** support for multi-GPU and multi-node setups

### 🔧 Development Tools
- **Pre-configured IDE** with AI-specific extensions and settings
- **Version control** with Git LFS support for large model files
- **Container-based development** with GPU passthrough support
- **Package management** with conda environments and offline repositories
- **Enhanced model manager** with integrity verification and snapshots

### 🚀 CI/CD Pipeline
- **Automated testing** for AI models and data pipelines
- **Model validation** with performance and accuracy testing
- **Pre-commit hooks** for code quality and security
- **Local deployment** pipeline with monitoring and logging
- **Container orchestration** for scalable deployments

### 📚 Offline-First
- **Cached models** and datasets for offline development
- **Local documentation** server with searchable AI references
- **Offline package repositories** for Python libraries
- **No internet required** for core development tasks
- **Enhanced documentation system** with interactive tutorials

### 🔧 System Management
- **Migration system** for seamless updates and feature additions
- **System monitoring** with comprehensive GPU and resource tracking
- **Container orchestration** for isolated development environments
- **Automated diagnostics** with repair recommendations
- **Performance optimization** tools and monitoring

## 📋 Installation Options

### Standard Installation
```bash
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash
```

### Advanced Installation
```bash
# Clone repository first
git clone https://github.com/mitkox/omarchy-ai.git
cd omarchy-ai

# Run system validation
./validate-system.sh

# Install with custom options
./install.sh
```

### Resume Installation
```bash
# Resume from last checkpoint
./install.sh --resume

# Rollback installation
./install.sh --rollback
```

## 🛠️ Essential Commands

### System Management
```bash
omarchy-ai-doctor           # Run comprehensive diagnostics
omarchy-ai-repair           # Automatic issue repair
omarchy-ai-test             # Run full test suite
omarchy-ai-test --quick     # Quick health check
```

### Environment Management
```bash
ai-env                      # Activate AI development environment
ai-workspace               # Navigate to AI workspace
conda deactivate           # Exit environment
```

### Model Management
```bash
model-download <model-id>   # Download from Hugging Face
model-list                 # List downloaded models
model-info <model-id>      # Show model details
model-verify <model-id>    # Verify model integrity
model-snapshot <model> <tag> # Create model snapshot
model-serve                # Start model API server
model-cleanup              # Clean up cache
```

### Development Tools
```bash
jupyter-ai                 # Start Jupyter Lab in AI workspace
mlflow-ui                  # MLflow experiment tracking
tensorboard-ai             # TensorBoard visualization  
docs-serve                 # Start documentation server
gpu-monitor                # Monitor GPU usage
```

### Container & CI/CD
```bash
containers start           # Start container services
containers stop            # Stop container services
ai-test                    # Run AI pipeline tests
ai-lint                    # Code quality checks
ai-format                  # Format code
```

## 🏗️ Project Structure

```
~/ai-workspace/
├── projects/          # AI project workspace
├── models/            # Downloaded AI models
│   ├── huggingface/   # Hugging Face models
│   ├── gguf/          # GGUF models for llama.cpp
│   ├── pytorch/       # PyTorch models
│   └── snapshots/     # Model version snapshots
├── datasets/          # Training and evaluation datasets
│   ├── raw/           # Original datasets
│   ├── processed/     # Processed datasets
│   └── splits/        # Train/test splits
├── experiments/       # ML experiment tracking
├── notebooks/         # Jupyter notebooks
├── logs/              # Application logs
├── cache/             # Temporary cache files
├── tools/             # Management scripts
└── docs/              # Offline documentation
```

## 💻 Hardware Requirements

### Minimum
- CPU: Modern x86_64 processor (4+ cores recommended)
- RAM: 16GB
- Storage: 100GB free space
- GPU: Optional (CPU inference fallback available)

### Recommended
- CPU: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- RAM: 32GB+
- Storage: 500GB+ NVMe SSD
- GPU: NVIDIA RTX 3080/4080 or better (8GB+ VRAM)

### Optimal
- CPU: 16+ cores (Intel i9/AMD Ryzen 9 or better)
- RAM: 64GB+
- Storage: 1TB+ NVMe SSD
- GPU: NVIDIA RTX 4090 or better (24GB+ VRAM)

## 🎯 Usage Examples

### Quick AI Chat
```bash
# Download and chat with a model
model-download microsoft/DialoGPT-medium
llama-chat
```

### ML Experiment Tracking
```bash
# Start MLflow
mlflow-ui

# Track experiments in Python
import mlflow
mlflow.start_run()
mlflow.log_metric("accuracy", 0.95)
mlflow.end_run()
```

### Model Serving
```bash
# Start model API server
model-serve --model microsoft/DialoGPT-medium --port 8000

# Test API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

### Container Development
```bash
# Start AI development container
containers start ai-jupyter

# Access container
docker exec -it ai-jupyter bash
```

## 🔧 Troubleshooting

### Quick Fixes
```bash
# System not working? Run diagnostics
omarchy-ai-doctor

# Found issues? Try automatic repair
omarchy-ai-repair

# Still having problems? Check logs
tail -f ~/.omarchy-ai-install.log
```

### Common Issues

**Environment activation fails:**
```bash
source ~/.bashrc
conda activate ai-dev
```

**GPU not detected:**
```bash
sudo pacman -S nvidia nvidia-utils
sudo reboot
```

**Model downloads failing:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
model-download microsoft/DialoGPT-small
```

**Jupyter won't start:**
```bash
conda activate ai-dev
pip install --force-reinstall jupyter jupyterlab
```

📖 **Need more help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) for detailed solutions.

## 🎮 Example Projects

### 1. Personal AI Assistant
- Download conversational model
- Create chat interface with memory
- Deploy as local web service
- Add voice input/output

### 2. Document Q&A System  
- Load document datasets
- Build semantic search with embeddings
- Create question-answering pipeline
- Deploy with FastAPI

### 3. Image Analysis Pipeline
- Set up computer vision models
- Create batch processing pipeline  
- Add model performance monitoring
- Containerize for deployment

### 4. Distributed Training Setup
- Configure multi-GPU training
- Set up experiment tracking
- Implement model checkpointing
- Add performance profiling

## 🧪 Testing Your Installation

### Quick Health Check
```bash
omarchy-ai-test --quick
```

### Full Test Suite  
```bash
omarchy-ai-test
```

### Performance Benchmarks
```bash
omarchy-ai-test --performance
omarchy-ai-doctor --performance
```

### GPU Testing
```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

## 📊 Monitoring & Optimization

### Resource Monitoring
```bash
gpu-monitor              # Real-time GPU stats
htop                     # CPU and memory usage
iotop                    # Disk I/O monitoring
```

### Performance Optimization
```bash
# Set CPU thread limits for optimal performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Configure GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory optimization
export PYTHONMALLOC=malloc
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use GitHub issues for bugs and feature requests
2. **Submit PRs**: Follow our contribution guidelines
3. **Improve Documentation**: Help make setup easier for everyone
4. **Share Examples**: Contribute example projects and tutorials
5. **Test & Feedback**: Try new features and provide feedback

### Development Setup
```bash
git clone https://github.com/mitkox/omarchy-ai.git
cd omarchy-ai
git checkout -b feature/your-feature

# Make changes and test
./bin/omarchy-ai-test

# Submit PR
```

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 15 minutes
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common issues  
- **[Installation Guide](INSTALLATION-GUIDE.md)** - Detailed installation instructions
- **[Product Requirements](PRD.md)** - Project vision and roadmap
- **API Documentation** - Available via `docs-serve`

## 🌟 Community

- **GitHub Discussions**: Ask questions and share projects
- **Issues**: Report bugs and request features  
- **Wiki**: Community-contributed guides and tips
- **Examples**: Share your AI projects and setups

## 📄 License

Omarchy AI is released under the [MIT License](https://opensource.org/licenses/MIT).

## 🙏 Acknowledgments

- Built on the excellent [Omarchy](https://github.com/basecamp/omarchy) foundation
- Powered by the amazing open-source AI community
- Thanks to all contributors and users who make this project possible

---

**Ready to start your AI development journey?** 

```bash
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash
```

*Star ⭐ this repo if you find it useful!*

