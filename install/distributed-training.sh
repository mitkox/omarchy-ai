# Distributed Training Setup for Omarchy AI
# Adds support for multi-GPU and distributed training

# Install distributed training dependencies
pip install \
  torch-distributed \
  horovod \
  deepspeed \
  accelerate \
  torch-elastic \
  ray[tune] \
  optuna \
  wandb \
  neptune-client

# Install communication libraries
yay -S --noconfirm --needed \
  openmpi \
  nccl \
  rdma-core

# Create distributed training configuration
mkdir -p ~/.config/distributed-training

cat > ~/.config/distributed-training/config.yaml << 'EOF'
# Distributed Training Configuration

cluster:
  type: local  # local, slurm, kubernetes
  nodes: 1
  gpus_per_node: auto  # auto-detect or specify number
  
communication:
  backend: nccl  # nccl, mpi, gloo
  timeout: 30  # minutes
  
training:
  mixed_precision: true
  gradient_clipping: 1.0
  accumulation_steps: 1
  
monitoring:
  wandb_project: omarchy-ai
  log_frequency: 100
  save_frequency: 1000
  
optimization:
  zero_optimization: true
  cpu_offload: false
  nvme_offload: false
EOF

# Create distributed training launcher script
cat > ~/ai-workspace/tools/distributed-train.py << 'EOF'
#!/usr/bin/env python3
"""
Distributed Training Launcher for Omarchy AI
Supports multi-GPU and multi-node training
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "~/.config/distributed-training/config.yaml"):
    """Load distributed training configuration."""
    config_path = Path(config_path).expanduser()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def detect_available_gpus():
    """Detect available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def setup_accelerate_config():
    """Setup accelerate configuration."""
    config = load_config()
    
    accelerate_config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU" if detect_available_gpus() > 1 else "NO",
        "mixed_precision": "fp16" if config["training"]["mixed_precision"] else "no",
        "use_cpu": False,
        "num_processes": detect_available_gpus(),
        "gpu_ids": "all",
        "machine_rank": 0,
        "num_machines": config["cluster"]["nodes"],
        "main_process_ip": "localhost",
        "main_process_port": 29500,
        "downcast_bf16": "no"
    }
    
    return accelerate_config

def launch_training(script_path: str, script_args: list):
    """Launch distributed training."""
    logger = setup_logging()
    config = load_config()
    
    # Detect available resources
    num_gpus = detect_available_gpus()
    if num_gpus == 0:
        logger.warning("No GPUs detected, falling back to CPU training")
        # Run on CPU
        cmd = [sys.executable, script_path] + script_args
    elif num_gpus == 1:
        logger.info("Single GPU detected, running single-GPU training")
        cmd = [sys.executable, script_path] + script_args
    else:
        logger.info(f"Multiple GPUs detected ({num_gpus}), launching distributed training")
        
        # Use accelerate for multi-GPU training
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "--mixed_precision", "fp16" if config["training"]["mixed_precision"] else "no",
            script_path
        ] + script_args
    
    logger.info(f"Launching command: {' '.join(cmd)}")
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    env["NCCL_DEBUG"] = "INFO"
    
    # Launch training
    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def create_training_template():
    """Create a distributed training template."""
    template_path = Path("~/ai-workspace/templates/distributed-training").expanduser()
    template_path.mkdir(parents=True, exist_ok=True)
    
    # Create training script template
    training_script = template_path / "train.py"
    with open(training_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Distributed Training Template for Omarchy AI
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
import logging
from pathlib import Path

def setup_logging():
    """Setup logging."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, optimizer, accelerator):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0 and accelerator.is_main_process:
            logging.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    return total_loss / len(dataloader)

def validate(model, dataloader, accelerator):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb-project', type=str, default='omarchy-ai')
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    logger = setup_logging()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb on main process
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Create dummy dataset (replace with your actual dataset)
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    logger.info(f"Starting training on {accelerator.num_processes} processes")
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, accelerator)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, accelerator)
        
        # Log metrics
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            save_path = f'checkpoint_epoch_{epoch + 1}.pt'
            accelerator.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint: {save_path}")
    
    if accelerator.is_main_process:
        logger.info("Training completed!")
        wandb.finish()

if __name__ == "__main__":
    main()
''')
    
    # Create requirements file
    requirements_file = template_path / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write('''torch
torchvision
accelerate
wandb
numpy
scikit-learn
''')
    
    print(f"Created distributed training template at: {template_path}")

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Launcher")
    parser.add_argument('script', help='Training script to run')
    parser.add_argument('script_args', nargs='*', help='Arguments for training script')
    parser.add_argument('--create-template', action='store_true', 
                       help='Create distributed training template')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_training_template()
        return
    
    if not os.path.exists(args.script):
        print(f"Error: Training script not found: {args.script}")
        sys.exit(1)
    
    launch_training(args.script, args.script_args)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/distributed-train.py

# Create accelerate configuration setup script
cat > ~/ai-workspace/tools/setup-accelerate.py << 'EOF'
#!/usr/bin/env python3
"""
Setup accelerate configuration for distributed training
"""

import json
import os
from pathlib import Path
import torch

def setup_accelerate_config():
    """Setup accelerate configuration based on available hardware."""
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU" if num_gpus > 1 else "NO",
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "fp16" if num_gpus > 0 else "no",
        "num_machines": 1,
        "num_processes": num_gpus if num_gpus > 0 else 1,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": num_gpus == 0
    }
    
    # Create accelerate config directory
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = config_dir / "default_config.yaml"
    
    yaml_content = f"""compute_environment: LOCAL_MACHINE
distributed_type: {'MULTI_GPU' if num_gpus > 1 else 'NO'}
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: {'fp16' if num_gpus > 0 else 'no'}
num_machines: 1
num_processes: {num_gpus if num_gpus > 0 else 1}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: {str(num_gpus == 0).lower()}
"""
    
    with open(config_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Accelerate configuration saved to: {config_file}")
    print(f"Detected {num_gpus} GPU(s)")
    print(f"Distributed type: {'MULTI_GPU' if num_gpus > 1 else 'NO'}")
    print(f"Mixed precision: {'fp16' if num_gpus > 0 else 'no'}")

if __name__ == "__main__":
    setup_accelerate_config()
EOF

chmod +x ~/ai-workspace/tools/setup-accelerate.py

# Setup accelerate configuration
python ~/ai-workspace/tools/setup-accelerate.py

# Add distributed training aliases
cat >> ~/.bashrc << 'EOF'

# Distributed Training Aliases
alias distributed-train='python ~/ai-workspace/tools/distributed-train.py'
alias setup-accelerate='python ~/ai-workspace/tools/setup-accelerate.py'
alias train-template='python ~/ai-workspace/tools/distributed-train.py --create-template'
EOF

echo "Distributed training setup complete!"
echo "Available commands:"
echo "  distributed-train <script.py> [args] - Launch distributed training"
echo "  setup-accelerate                     - Setup accelerate configuration"
echo "  train-template                       - Create distributed training template"
