# Container-based Development Environment for Omarchy AI
# Provides isolated, reproducible AI development environments

# Install container tools
yay -S --noconfirm --needed \
  podman podman-compose \
  buildah skopeo \
  crun \
  slirp4netns \
  fuse-overlayfs

# Install additional container tools
pip install \
  docker-compose \
  podman-py \
  testcontainers

# Configure podman for rootless operation
if ! grep -q "^$(whoami):" /etc/subuid; then
    echo "$(whoami):100000:65536" | sudo tee -a /etc/subuid
fi

if ! grep -q "^$(whoami):" /etc/subgid; then
    echo "$(whoami):100000:65536" | sudo tee -a /etc/subgid
fi

# Create container configurations directory
mkdir -p ~/ai-workspace/containers/{images,compose,volumes}

# Create base AI development Dockerfile
cat > ~/ai-workspace/containers/images/Dockerfile.ai-base << 'EOF'
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git git-lfs \
    curl wget \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libhdf5-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorflow[and-cuda] \
    transformers datasets tokenizers accelerate \
    numpy pandas scikit-learn matplotlib seaborn \
    jupyter jupyterlab \
    mlflow wandb \
    fastapi uvicorn \
    pytest black flake8 isort mypy

# Install llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    make LLAMA_CUBLAS=1 && \
    cp main /usr/local/bin/llama-main && \
    cp server /usr/local/bin/llama-server

# Create workspace directory
WORKDIR /workspace

# Set up non-root user
RUN useradd -m -s /bin/bash aidev && \
    echo "aidev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER aidev

# Set default command
CMD ["/bin/bash"]
EOF

# Create PyTorch development container
cat > ~/ai-workspace/containers/images/Dockerfile.pytorch << 'EOF'
FROM ai-base:latest

# Install additional PyTorch ecosystem packages
RUN pip3 install --no-cache-dir \
    torchtext \
    torchdata \
    torchaudio \
    torchvision \
    pytorch-lightning \
    torchmetrics \
    timm \
    albumentations \
    opencv-python \
    pillow

# Install research tools
RUN pip3 install --no-cache-dir \
    tensorboard \
    wandb \
    optuna \
    ray[tune] \
    hydra-core \
    omegaconf

WORKDIR /workspace
EOF

# Create TensorFlow development container
cat > ~/ai-workspace/containers/images/Dockerfile.tensorflow << 'EOF'
FROM ai-base:latest

# Install TensorFlow ecosystem
RUN pip3 install --no-cache-dir \
    tensorflow-datasets \
    tensorflow-hub \
    tensorflow-probability \
    tensorflow-addons \
    tf-agents \
    keras-tuner

# Install additional tools
RUN pip3 install --no-cache-dir \
    tensorboard \
    wandb \
    apache-beam \
    tfx

WORKDIR /workspace
EOF

# Create Hugging Face development container
cat > ~/ai-workspace/containers/images/Dockerfile.huggingface << 'EOF'
FROM ai-base:latest

# Install Hugging Face ecosystem
RUN pip3 install --no-cache-dir \
    transformers[torch,tf] \
    datasets \
    tokenizers \
    accelerate \
    diffusers \
    gradio \
    streamlit \
    evaluate \
    peft \
    bitsandbytes

# Install additional NLP tools
RUN pip3 install --no-cache-dir \
    spacy \
    nltk \
    sentence-transformers \
    faiss-cpu \
    chromadb

# Download spaCy models
RUN python3 -m spacy download en_core_web_sm

WORKDIR /workspace
EOF

# Create LLM development container
cat > ~/ai-workspace/containers/images/Dockerfile.llm << 'EOF'
FROM ai-base:latest

# Install LLM tools
RUN pip3 install --no-cache-dir \
    langchain \
    langchain-community \
    openai \
    anthropic \
    ollama \
    llama-cpp-python[server] \
    guidance \
    outlines

# Install vector databases
RUN pip3 install --no-cache-dir \
    chromadb \
    weaviate-client \
    qdrant-client \
    pinecone-client

# Install additional tools
RUN pip3 install --no-cache-dir \
    tiktoken \
    sentence-transformers \
    rank-bm25

WORKDIR /workspace
EOF

# Create development environment docker-compose
cat > ~/ai-workspace/containers/compose/docker-compose.yml << 'EOF'
version: '3.8'

services:
  ai-jupyter:
    build:
      context: ../images
      dockerfile: Dockerfile.ai-base
    container_name: ai-jupyter
    ports:
      - "8888:8888"
      - "6006:6006"  # TensorBoard
    volumes:
      - ~/ai-workspace:/workspace
      - ~/.ssh:/home/aidev/.ssh:ro
      - ~/.gitconfig:/home/aidev/.gitconfig:ro
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - NVIDIA_VISIBLE_DEVICES=all
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ai-pytorch:
    build:
      context: ../images
      dockerfile: Dockerfile.pytorch
    container_name: ai-pytorch
    volumes:
      - ~/ai-workspace:/workspace
      - ~/.ssh:/home/aidev/.ssh:ro
      - ~/.gitconfig:/home/aidev/.gitconfig:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ai-tensorflow:
    build:
      context: ../images
      dockerfile: Dockerfile.tensorflow
    container_name: ai-tensorflow
    volumes:
      - ~/ai-workspace:/workspace
      - ~/.ssh:/home/aidev/.ssh:ro
      - ~/.gitconfig:/home/aidev/.gitconfig:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ai-huggingface:
    build:
      context: ../images
      dockerfile: Dockerfile.huggingface
    container_name: ai-huggingface
    ports:
      - "7860:7860"  # Gradio
      - "8501:8501"  # Streamlit
    volumes:
      - ~/ai-workspace:/workspace
      - ~/.ssh:/home/aidev/.ssh:ro
      - ~/.gitconfig:/home/aidev/.gitconfig:ro
      - ~/.cache/huggingface:/home/aidev/.cache/huggingface
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_HOME=/home/aidev/.cache/huggingface
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ai-llm:
    build:
      context: ../images
      dockerfile: Dockerfile.llm
    container_name: ai-llm
    ports:
      - "8080:8080"  # llama.cpp server
      - "8000:8000"  # FastAPI
    volumes:
      - ~/ai-workspace:/workspace
      - ~/.ssh:/home/aidev/.ssh:ro
      - ~/.gitconfig:/home/aidev/.gitconfig:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # Supporting services
  mlflow:
    image: python:3.11-slim
    container_name: ai-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ~/ai-workspace/mlruns:/mlruns
    command: >
      sh -c "pip install mlflow &&
             mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlruns"

  minio:
    image: minio/minio:latest
    container_name: ai-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ~/ai-workspace/minio-data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"

volumes:
  ai-workspace:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ~/ai-workspace
EOF

# Create container management script
cat > ~/ai-workspace/tools/container-manager.py << 'EOF'
#!/usr/bin/env python3
"""
Container Management Tool for Omarchy AI
Manages development containers with GPU support
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
import docker
import json

class ContainerManager:
    def __init__(self):
        self.containers_dir = Path.home() / "ai-workspace" / "containers"
        self.compose_file = self.containers_dir / "compose" / "docker-compose.yml"
        
    def build_images(self):
        """Build all container images."""
        print("Building container images...")
        
        images_dir = self.containers_dir / "images"
        
        # Build base image first
        self._build_image("ai-base", images_dir / "Dockerfile.ai-base")
        
        # Build specialized images
        specialized_images = {
            "ai-pytorch": "Dockerfile.pytorch",
            "ai-tensorflow": "Dockerfile.tensorflow", 
            "ai-huggingface": "Dockerfile.huggingface",
            "ai-llm": "Dockerfile.llm"
        }
        
        for image_name, dockerfile in specialized_images.items():
            self._build_image(image_name, images_dir / dockerfile)
    
    def _build_image(self, name: str, dockerfile: Path):
        """Build a single container image."""
        print(f"Building {name}...")
        
        cmd = [
            "podman", "build",
            "-t", name,
            "-f", str(dockerfile),
            str(dockerfile.parent)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully built {name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build {name}: {e}")
    
    def start_service(self, service: str = None):
        """Start container services."""
        if service:
            print(f"Starting {service}...")
            cmd = ["podman-compose", "-f", str(self.compose_file), "up", "-d", service]
        else:
            print("Starting all services...")
            cmd = ["podman-compose", "-f", str(self.compose_file), "up", "-d"]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.containers_dir / "compose")
            print("Services started successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start services: {e}")
    
    def stop_service(self, service: str = None):
        """Stop container services."""
        if service:
            print(f"Stopping {service}...")
            cmd = ["podman-compose", "-f", str(self.compose_file), "down", service]
        else:
            print("Stopping all services...")
            cmd = ["podman-compose", "-f", str(self.compose_file), "down"]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.containers_dir / "compose")
            print("Services stopped successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop services: {e}")
    
    def list_services(self):
        """List running services."""
        cmd = ["podman-compose", "-f", str(self.compose_file), "ps"]
        subprocess.run(cmd, cwd=self.containers_dir / "compose")
    
    def exec_container(self, service: str, command: str = "/bin/bash"):
        """Execute command in container."""
        cmd = ["podman-compose", "-f", str(self.compose_file), "exec", service, command]
        subprocess.run(cmd, cwd=self.containers_dir / "compose")
    
    def logs(self, service: str):
        """Show container logs."""
        cmd = ["podman-compose", "-f", str(self.compose_file), "logs", "-f", service]
        subprocess.run(cmd, cwd=self.containers_dir / "compose")
    
    def create_project_container(self, project_name: str, base_image: str = "ai-base"):
        """Create a project-specific container."""
        project_dir = Path.home() / "ai-workspace" / "projects" / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project Dockerfile
        dockerfile_content = f"""
FROM {base_image}:latest

# Project-specific dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set up project workspace
WORKDIR /workspace/{project_name}

# Copy project files
COPY . /workspace/{project_name}/

# Set default command
CMD ["/bin/bash"]
"""
        
        with open(project_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt if it doesn't exist
        requirements_file = project_dir / "requirements.txt"
        if not requirements_file.exists():
            with open(requirements_file, 'w') as f:
                f.write("# Project-specific requirements\n")
        
        print(f"Created container setup for project: {project_name}")
        print(f"Edit {requirements_file} to add project dependencies")
        print(f"Build with: podman build -t {project_name} {project_dir}")

def main():
    parser = argparse.ArgumentParser(description="Container Management for AI Development")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build container images")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument("service", nargs="?", help="Service to start")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop services")
    stop_parser.add_argument("service", nargs="?", help="Service to stop")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List running services")
    
    # Exec command
    exec_parser = subparsers.add_parser("exec", help="Execute command in container")
    exec_parser.add_argument("service", help="Service name")
    exec_parser.add_argument("command", nargs="?", default="/bin/bash", help="Command to execute")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show container logs")
    logs_parser.add_argument("service", help="Service name")
    
    # Create project command
    create_parser = subparsers.add_parser("create-project", help="Create project container")
    create_parser.add_argument("project_name", help="Project name")
    create_parser.add_argument("--base-image", default="ai-base", help="Base image")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ContainerManager()
    
    if args.command == "build":
        manager.build_images()
    elif args.command == "start":
        manager.start_service(args.service)
    elif args.command == "stop":
        manager.stop_service(args.service)
    elif args.command == "list":
        manager.list_services()
    elif args.command == "exec":
        manager.exec_container(args.service, args.command)
    elif args.command == "logs":
        manager.logs(args.service)
    elif args.command == "create-project":
        manager.create_project_container(args.project_name, args.base_image)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/container-manager.py

# Add container aliases
cat >> ~/.bashrc << 'EOF'

# Container Development Aliases
alias containers='python ~/ai-workspace/tools/container-manager.py'
alias ai-jupyter='containers start ai-jupyter'
alias ai-pytorch='containers exec ai-pytorch'
alias ai-tensorflow='containers exec ai-tensorflow'
alias ai-huggingface='containers exec ai-huggingface'
alias ai-llm='containers exec ai-llm'
alias containers-build='containers build'
alias containers-start='containers start'
alias containers-stop='containers stop'
alias containers-list='containers list'
EOF

echo "Container-based development environment setup complete!"
echo "Available commands:"
echo "  containers build              - Build all container images"
echo "  containers start [service]    - Start container services"
echo "  containers stop [service]     - Stop container services"
echo "  containers list               - List running services"
echo "  containers exec <service>     - Access container shell"
echo "  ai-jupyter                    - Start Jupyter container"
echo "  ai-pytorch                    - Access PyTorch container"
echo "  ai-tensorflow                 - Access TensorFlow container"
echo "  ai-huggingface                - Access Hugging Face container"
echo "  ai-llm                        - Access LLM container"
