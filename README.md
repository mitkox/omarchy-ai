# Omarchy AI

Turn a fresh Arch installation into a fully-configured, beautiful, and modern AI development system based on Hyprland by running a single command. Built on the foundation of Omarchy, Omarchy AI is specifically designed for AI engineers and researchers who need a complete, local-first development environment.

Unlike cloud-based AI development, Omarchy AI provides:
- **Local AI inference** with llama.cpp and CUDA GPU support
- **Offline-first development** with cached models and documentation
- **Integrated CI/CD** for AI model testing and deployment
- **Model management** for downloading, versioning, and serving AI models
- **Privacy-focused** development with no cloud dependencies

Read more about the original Omarchy at [omarchy.org](https://omarchy.org).

## Features

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
- **Automated updates** with rollback capabilities

## Installation

### Prerequisites
- Fresh Arch Linux installation
- NVIDIA GPU (recommended for optimal performance)
- At least 16GB RAM (32GB recommended)
- 500GB+ storage for models and datasets

### Quick Start
```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/refs/heads/main/boot.sh | bash
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/mitkox/omarchy-ai.git
cd omarchy-ai

# Run the installation
./install.sh
```

The installation will:
1. Set up the base Omarchy system (Hyprland, themes, etc.)
2. Install AI development tools (Python, PyTorch, TensorFlow)
3. Configure llama.cpp with CUDA support
4. Set up model management and CI/CD tools
5. Download offline documentation and references

## Usage

### AI Development Workflow
```bash
# Activate AI environment
ai-env

# Navigate to workspace
ai-workspace

# Start Jupyter Lab
jupyter-ai

# Download and manage models
model-download Qwen/Qwen3-0.6B
model-list
model-serve

# Initialize new AI project
ai-init my-ai-project
```

### Local Model Inference
```bash
# Interactive chat with local model
llama-chat

# Start llama.cpp server
llama-start

# Monitor GPU usage
gpu-monitor

# Test CUDA setup
gpu-test
```

### CI/CD and Testing
```bash
# Run tests
ai-test

# Check code quality
ai-lint
ai-security

# Format code
ai-format
```

### Offline Documentation
```bash
# Start documentation server
docs-serve

# Search documentation
docs-search "transformer"

# Update documentation
docs-update
```

## Project Structure

```
~/ai-workspace/
├── projects/          # AI project workspace
├── models/            # Downloaded AI models
│   ├── huggingface/   # Hugging Face models
│   ├── gguf/          # GGUF models for llama.cpp
│   └── custom/        # Custom trained models
├── datasets/          # Training and evaluation datasets
├── experiments/       # ML experiment tracking
├── notebooks/         # Jupyter notebooks
├── tools/             # Management scripts
└── docs/              # Offline documentation
```

## Hardware Requirements

### Minimum
- CPU: Modern x86_64 processor
- RAM: 16GB
- Storage: 100GB free space
- GPU: Optional (CPU inference fallback)

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

## Available Commands

### Model Management
- `model-download <model-id>` - Download model from Hugging Face
- `model-list` - List downloaded models
- `model-info <model-id>` - Show model information
- `model-serve` - Start model API server
- `model-cleanup` - Clean up model cache
- `model-verify <model-id>` - Verify model integrity
- `model-snapshot <model-id> <tag>` - Create model snapshot

### Development
- `ai-init <project-name>` - Initialize new AI project
- `ai-env` - Activate AI conda environment
- `ai-workspace` - Navigate to AI workspace
- `jupyter-ai` - Start Jupyter Lab in AI workspace
- `distributed-train <script.py>` - Launch distributed training

### GPU and Performance
- `gpu-monitor` - Monitor GPU usage with detailed stats
- `gpu-monitor --continuous` - Continuous GPU monitoring
- `gpu-benchmark` - Run GPU performance benchmark
- `gpu-test` - Test CUDA functionality

### Containers
- `containers build` - Build all container images
- `containers start [service]` - Start container services
- `containers stop [service]` - Stop container services
- `containers list` - List running services
- `ai-jupyter` - Start Jupyter container
- `ai-pytorch` - Access PyTorch container

### Documentation
- `docs-serve` - Start offline documentation server
- `docs-build` - Build documentation
- `docs-update` - Update framework documentation
- `docs-search <query>` - Search documentation

### System Management
- `system-update` - Full system update including migrations
- `run-migrations` - Run pending migrations
- `create-migration "description"` - Create new migration
- `list-migrations` - List migration status

### llama.cpp
- `llama-chat` - Interactive chat with local model
- `llama-download` - Download GGUF models
- `llama-start/stop` - Control llama.cpp server

## Troubleshooting

### GPU Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Test CUDA installation
gpu-test

# Verify GPU support in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Loading Issues
```bash
# Check model cache
model-list

# Verify model integrity
model-verify <model-id>

# Clean up corrupted cache
model-cleanup
```

### Performance Issues
```bash
# Monitor system resources
htop

# Check GPU utilization
gpu-monitor

# Benchmark performance
gpu-benchmark
```

## Contributing

Contributions are welcome! Please read the [PRD](PRD.md) for project requirements and development guidelines.

## License

Omarchy AI is released under the [MIT License](https://opensource.org/licenses/MIT).

