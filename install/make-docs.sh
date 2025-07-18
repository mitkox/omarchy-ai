# Enhanced Offline Documentation System for Omarchy AI
# Creates a comprehensive local documentation server with AI-specific content

# Install documentation tools
pip install \
  mkdocs mkdocs-material \
  mkdocs-awesome-pages-plugin \
  mkdocs-mermaid2-plugin \
  mkdocs-jupyter \
  mkdocs-macros-plugin \
  sphinx sphinx-rtd-theme \
  pydoctor \
  portray \
  pdoc3 \
  jupyterbook

# Install additional documentation dependencies
yay -S --noconfirm --needed \
  pandoc \
  hugo \
  zeal \
  devhelp \
  devdocs-desktop

# Create documentation structure
mkdir -p ~/ai-workspace/docs/{frameworks,models,tutorials,api,datasets}
mkdir -p ~/ai-workspace/docs/site

# Download offline documentation for major AI frameworks
echo "Setting up offline documentation..."

# Create documentation download script
cat > ~/ai-workspace/tools/download-docs.py << 'EOF'
#!/usr/bin/env python3
"""
Download and setup offline documentation for AI frameworks
"""

import os
import subprocess
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import json
import requests
from tqdm import tqdm

DOCS_DIR = Path.home() / "ai-workspace" / "docs"
FRAMEWORKS_DIR = DOCS_DIR / "frameworks"

def download_file(url: str, filepath: Path, description: str = "Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

def setup_pytorch_docs():
    """Download PyTorch documentation."""
    print("Setting up PyTorch documentation...")
    
    pytorch_dir = FRAMEWORKS_DIR / "pytorch"
    pytorch_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone PyTorch documentation
    if not (pytorch_dir / ".git").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/pytorch/pytorch.git",
            str(pytorch_dir / "source")
        ], check=True)
    
    # Build documentation if sphinx is available
    try:
        docs_source = pytorch_dir / "source" / "docs"
        if docs_source.exists():
            subprocess.run([
                "sphinx-build", "-b", "html",
                str(docs_source / "source"),
                str(pytorch_dir / "html")
            ], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not build PyTorch docs")

def setup_transformers_docs():
    """Download Transformers documentation."""
    print("Setting up Transformers documentation...")
    
    transformers_dir = FRAMEWORKS_DIR / "transformers"
    transformers_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone Transformers documentation
    if not (transformers_dir / ".git").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/huggingface/transformers.git",
            str(transformers_dir / "source")
        ], check=True)

def setup_tensorflow_docs():
    """Setup TensorFlow documentation."""
    print("Setting up TensorFlow documentation...")
    
    tf_dir = FRAMEWORKS_DIR / "tensorflow"
    tf_dir.mkdir(parents=True, exist_ok=True)
    
    # Download TensorFlow docs
    if not (tf_dir / ".git").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/tensorflow/docs.git",
            str(tf_dir / "source")
        ], check=True)

def setup_langchain_docs():
    """Setup LangChain documentation."""
    print("Setting up LangChain documentation...")
    
    langchain_dir = FRAMEWORKS_DIR / "langchain"
    langchain_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone LangChain documentation
    if not (langchain_dir / ".git").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/langchain-ai/langchain.git",
            str(langchain_dir / "source")
        ], check=True)

def setup_custom_docs():
    """Setup custom documentation structure."""
    print("Setting up custom documentation...")
    
    # Create MkDocs configuration
    mkdocs_config = DOCS_DIR / "mkdocs.yml"
    with open(mkdocs_config, 'w') as f:
        f.write("""
site_name: Omarchy AI Documentation
site_description: Local AI Development Documentation
site_author: Omarchy AI

theme:
  name: material
  palette:
    - scheme: default
      primary: blue
      accent: orange
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: blue
      accent: orange
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.share
    - content.code.annotate

plugins:
  - search
  - awesome-pages
  - mermaid2
  - mkdocs-jupyter
  - macros

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - attr_list
  - md_in_html

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
    - Configuration: getting-started/configuration.md
  - Frameworks:
    - PyTorch: frameworks/pytorch.md
    - TensorFlow: frameworks/tensorflow.md
    - Transformers: frameworks/transformers.md
    - LangChain: frameworks/langchain.md
  - Tutorials:
    - Model Training: tutorials/training.md
    - Distributed Training: tutorials/distributed.md
    - Model Deployment: tutorials/deployment.md
  - API Reference:
    - Model Manager: api/model-manager.md
    - GPU Monitor: api/gpu-monitor.md
    - CI/CD Tools: api/cicd.md
  - Examples:
    - Basic Examples: examples/basic.md
    - Advanced Examples: examples/advanced.md

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/basecamp/omarchy-ai
""")
    
    # Create main index page
    index_md = DOCS_DIR / "docs" / "index.md"
    index_md.parent.mkdir(parents=True, exist_ok=True)
    
    with open(index_md, 'w') as f:
        f.write("""
# Omarchy AI Documentation

Welcome to the comprehensive documentation for Omarchy AI - your complete local-first AI development environment.

## What is Omarchy AI?

Omarchy AI transforms a fresh Arch Linux installation into a fully-configured, beautiful, and modern AI development system. Built on the foundation of Omarchy, it provides everything you need for AI engineering and research with complete offline capabilities.

## Key Features

### 🤖 Local AI Development
- **Complete offline operation** - No cloud dependencies
- **GPU acceleration** with CUDA support
- **Model management** with version control
- **Distributed training** across multiple GPUs

### 🔧 Development Tools
- **Pre-configured environment** with all major AI frameworks
- **Integrated CI/CD** for AI model testing
- **Code quality tools** and automated testing
- **Jupyter notebooks** for interactive development

### 📚 Documentation
- **Offline documentation** for all major frameworks
- **Interactive tutorials** and examples
- **API references** for all tools
- **Best practices** and workflows

## Quick Start

1. **Installation**: Follow the [installation guide](getting-started/installation.md)
2. **Configuration**: Set up your [development environment](getting-started/configuration.md)
3. **First Model**: Try the [quick start tutorial](getting-started/quickstart.md)

## Framework Documentation

Access offline documentation for major AI frameworks:

- [PyTorch](frameworks/pytorch.md) - Deep learning framework
- [TensorFlow](frameworks/tensorflow.md) - Machine learning platform
- [Transformers](frameworks/transformers.md) - Hugging Face transformers
- [LangChain](frameworks/langchain.md) - LLM application framework

## Tools & Utilities

Explore the comprehensive toolkit:

- [Model Manager](api/model-manager.md) - Download and manage AI models
- [GPU Monitor](api/gpu-monitor.md) - Monitor GPU performance
- [Distributed Training](tutorials/distributed.md) - Scale training across GPUs
- [CI/CD Tools](api/cicd.md) - Automated testing and deployment

## Get Help

- Check the [tutorials](tutorials/training.md) for step-by-step guides
- Browse [examples](examples/basic.md) for practical implementations
- Review [API documentation](api/model-manager.md) for detailed references

---

*Built with ❤️ by the Omarchy AI community*
""")

def create_documentation_pages():
    """Create documentation pages."""
    docs_dir = DOCS_DIR / "docs"
    
    # Create getting started pages
    getting_started_dir = docs_dir / "getting-started"
    getting_started_dir.mkdir(parents=True, exist_ok=True)
    
    # Installation page
    with open(getting_started_dir / "installation.md", 'w') as f:
        f.write("""
# Installation Guide

## Prerequisites

- Fresh Arch Linux installation
- NVIDIA GPU (recommended)
- At least 16GB RAM
- 500GB+ storage

## Quick Installation

```bash
curl -fsSL https://raw.githubusercontent.com/your-repo/omarchy-ai/main/boot.sh | bash
```

## Manual Installation

```bash
git clone https://github.com/your-repo/omarchy-ai.git
cd omarchy-ai
./install.sh
```

## Post-Installation

1. Reboot your system
2. Activate AI environment: `ai-env`
3. Download your first model: `model-download microsoft/DialoGPT-medium`
4. Start developing: `ai-init my-project`
""")
    
    # Quick start page
    with open(getting_started_dir / "quickstart.md", 'w') as f:
        f.write("""
# Quick Start Guide

## Your First AI Project

### 1. Activate Environment
```bash
ai-env
```

### 2. Create Project
```bash
ai-init my-first-ai-project
cd ~/ai-workspace/projects/my-first-ai-project
```

### 3. Download Model
```bash
model-download microsoft/DialoGPT-medium
```

### 4. Start Jupyter
```bash
jupyter-ai
```

### 5. Train Model
```bash
make model-train
```

## Next Steps

- Explore [tutorials](../tutorials/training.md)
- Check [API documentation](../api/model-manager.md)
- Join the community discussions
""")

def main():
    """Main setup function."""
    print("Setting up offline documentation...")
    
    # Create directories
    FRAMEWORKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup framework documentation
    setup_pytorch_docs()
    setup_transformers_docs()
    setup_tensorflow_docs()
    setup_langchain_docs()
    
    # Setup custom documentation
    setup_custom_docs()
    create_documentation_pages()
    
    print("Documentation setup complete!")
    print(f"Documentation available at: {DOCS_DIR}")
    print("To serve documentation: docs-serve")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/download-docs.py

# Create documentation server script
cat > ~/ai-workspace/tools/docs-server.py << 'EOF'
#!/usr/bin/env python3
"""
Documentation Server for Omarchy AI
Serves offline documentation with search capabilities
"""

import argparse
import http.server
import socketserver
import webbrowser
import subprocess
import os
import threading
import time
from pathlib import Path

def serve_mkdocs(port: int = 8001):
    """Serve MkDocs documentation."""
    docs_dir = Path.home() / "ai-workspace" / "docs"
    os.chdir(docs_dir)
    
    try:
        subprocess.run(["mkdocs", "serve", "--dev-addr", f"0.0.0.0:{port}"], check=True)
    except KeyboardInterrupt:
        print("\nDocumentation server stopped")

def serve_static(directory: str, port: int = 8002):
    """Serve static documentation files."""
    os.chdir(directory)
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()
    
    with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
        print(f"Serving documentation at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDocumentation server stopped")

def open_browser(url: str, delay: int = 2):
    """Open browser after delay."""
    time.sleep(delay)
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(description="Documentation Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--static", help="Serve static files from directory")
    
    args = parser.parse_args()
    
    if args.static:
        if not args.no_browser:
            threading.Thread(
                target=open_browser,
                args=(f"http://localhost:{args.port}",),
                daemon=True
            ).start()
        serve_static(args.static, args.port)
    else:
        if not args.no_browser:
            threading.Thread(
                target=open_browser,
                args=(f"http://localhost:{args.port}",),
                daemon=True
            ).start()
        serve_mkdocs(args.port)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/docs-server.py

# Build initial documentation
python ~/ai-workspace/tools/download-docs.py

# Add documentation aliases
cat >> ~/.bashrc << 'EOF'

# Documentation Aliases
alias docs-serve='python ~/ai-workspace/tools/docs-server.py'
alias docs-build='cd ~/ai-workspace/docs && mkdocs build'
alias docs-update='python ~/ai-workspace/tools/download-docs.py'
alias docs-search='cd ~/ai-workspace/docs && grep -r'
EOF

echo "Enhanced documentation system setup complete!"
echo "Available commands:"
echo "  docs-serve  - Start documentation server"
echo "  docs-build  - Build documentation"
echo "  docs-update - Update framework documentation"
echo "  docs-search - Search documentation"
