# Model Management System for Omarchy AI
# Provides tools for downloading, versioning, and managing AI models

# Install model management dependencies
pip install \
  huggingface-hub \
  transformers \
  datasets \
  dvc \
  git-lfs \
  boto3 \
  minio \
  fsspec \
  aiofiles \
  aiohttp \
  tqdm \
  pydantic

# Initialize git-lfs for large file support
git lfs install

# Create model management directory structure
mkdir -p ~/ai-workspace/models/{huggingface,gguf,pytorch,tensorflow,onnx,custom}
mkdir -p ~/ai-workspace/datasets/{raw,processed,splits}
mkdir -p ~/ai-workspace/model-registry
mkdir -p ~/ai-workspace/model-cache

# Create model management configuration
cat > ~/.config/model-manager.yaml << 'EOF'
model_registry:
  local_path: /home/${USER}/ai-workspace/model-registry
  cache_path: /home/${USER}/ai-workspace/model-cache
  max_cache_size: 50GB
  
storage:
  backends:
    - name: local
      type: filesystem
      path: /home/${USER}/ai-workspace/models
    - name: huggingface
      type: huggingface_hub
      cache_dir: /home/${USER}/ai-workspace/models/huggingface
    - name: minio
      type: s3
      endpoint: http://localhost:9000
      access_key: minioadmin
      secret_key: minioadmin
      bucket: ai-models

model_categories:
  - name: language_models
    path: language-models
    formats: [pytorch, gguf, safetensors]
  - name: vision_models
    path: vision-models
    formats: [pytorch, onnx, tensorflow]
  - name: audio_models
    path: audio-models
    formats: [pytorch, onnx]
  - name: embedding_models
    path: embedding-models
    formats: [pytorch, safetensors]

offline_mode: true
auto_cleanup: true
verify_checksums: true
EOF

# Create model management CLI tool
cat > ~/ai-workspace/tools/model-manager.py << 'EOF'
#!/usr/bin/env python3
"""
Model Management CLI for Omarchy AI
Provides tools for downloading, versioning, and managing AI models
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import hashlib
from datetime import datetime
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
import torch

class ModelManager:
    def __init__(self, config_path: str = "~/.config/model-manager.yaml"):
        self.config_path = Path(config_path).expanduser()
        self.config = self.load_config()
        self.model_registry = Path(self.config["model_registry"]["local_path"])
        self.cache_path = Path(self.config["model_registry"]["cache_path"])
        self.models_path = Path(self.config["storage"]["backends"][0]["path"])
        
        # Create directories if they don't exist
        self.model_registry.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            sys.exit(1)
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information from registry."""
        registry_file = self.model_registry / f"{model_id.replace('/', '_')}.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_model_info(self, model_id: str, info: Dict):
        """Save model information to registry."""
        registry_file = self.model_registry / f"{model_id.replace('/', '_')}.json"
        with open(registry_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def download_huggingface_model(self, model_id: str, revision: str = "main") -> str:
        """Download model from Hugging Face Hub."""
        print(f"Downloading {model_id} from Hugging Face Hub...")
        
        try:
            # Download model files
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_path,
                local_dir=self.models_path / "huggingface" / model_id,
                local_dir_use_symlinks=False
            )
            
            # Save model info
            info = {
                "model_id": model_id,
                "source": "huggingface",
                "revision": revision,
                "path": str(model_path),
                "downloaded_at": datetime.now().isoformat(),
                "size": self.get_directory_size(model_path)
            }
            self.save_model_info(model_id, info)
            
            print(f"Model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    
    def download_gguf_model(self, repo_id: str, filename: str) -> str:
        """Download GGUF model file."""
        print(f"Downloading GGUF model {filename} from {repo_id}...")
        
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_path,
                local_dir=self.models_path / "gguf",
                local_dir_use_symlinks=False
            )
            
            # Save model info
            info = {
                "model_id": f"{repo_id}/{filename}",
                "source": "huggingface_gguf",
                "repo_id": repo_id,
                "filename": filename,
                "path": str(model_path),
                "downloaded_at": datetime.now().isoformat(),
                "size": os.path.getsize(model_path)
            }
            self.save_model_info(f"{repo_id}/{filename}", info)
            
            print(f"GGUF model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error downloading GGUF model: {e}")
            return None
    
    def list_models(self) -> List[Dict]:
        """List all downloaded models."""
        models = []
        for registry_file in self.model_registry.glob("*.json"):
            with open(registry_file, 'r') as f:
                models.append(json.load(f))
        return models
    
    def get_directory_size(self, path: str) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    
    def format_size(self, size: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def delete_model(self, model_id: str):
        """Delete a model and its registry entry."""
        info = self.get_model_info(model_id)
        if not info:
            print(f"Model {model_id} not found in registry")
            return
        
        model_path = Path(info["path"])
        if model_path.exists():
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
        
        # Remove registry entry
        registry_file = self.model_registry / f"{model_id.replace('/', '_')}.json"
        if registry_file.exists():
            registry_file.unlink()
        
        print(f"Model {model_id} deleted successfully")
    
    def verify_model(self, model_id: str) -> bool:
        """Verify model integrity."""
        info = self.get_model_info(model_id)
        if not info:
            print(f"Model {model_id} not found in registry")
            return False
        
        model_path = Path(info["path"])
        if not model_path.exists():
            print(f"Model path not found: {model_path}")
            return False
        
        # For PyTorch models, try to load them
        if info["source"] == "huggingface" and (model_path / "pytorch_model.bin").exists():
            try:
                torch.load(model_path / "pytorch_model.bin", map_location="cpu")
                print(f"Model {model_id} verification successful")
                return True
            except Exception as e:
                print(f"Model {model_id} verification failed: {e}")
                return False
        
        print(f"Model {model_id} basic verification passed")
        return True
    
    def cleanup_cache(self):
        """Clean up model cache."""
        cache_size = self.get_directory_size(self.cache_path)
        max_size = self.config["model_registry"]["max_cache_size"]
        
        # Convert max_size to bytes
        if max_size.endswith("GB"):
            max_size_bytes = int(max_size[:-2]) * 1024 * 1024 * 1024
        elif max_size.endswith("TB"):
            max_size_bytes = int(max_size[:-2]) * 1024 * 1024 * 1024 * 1024
        else:
            max_size_bytes = int(max_size)
        
        if cache_size > max_size_bytes:
            print(f"Cache size ({self.format_size(cache_size)}) exceeds limit ({max_size})")
            print("Cleaning up cache...")
            
            # Simple cleanup: remove oldest files
            cache_files = []
            for root, dirs, files in os.walk(self.cache_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    cache_files.append((file_path, os.path.getmtime(file_path)))
            
            cache_files.sort(key=lambda x: x[1])  # Sort by modification time
            
            for file_path, _ in cache_files:
                os.remove(file_path)
                cache_size -= os.path.getsize(file_path)
                if cache_size <= max_size_bytes:
                    break
            
            print("Cache cleanup completed")
        else:
            print(f"Cache size ({self.format_size(cache_size)}) is within limit")

def main():
    parser = argparse.ArgumentParser(description="AI Model Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="Model ID (e.g., microsoft/DialoGPT-medium)")
    download_parser.add_argument("--revision", default="main", help="Model revision/branch")
    download_parser.add_argument("--type", choices=["huggingface", "gguf"], default="huggingface", help="Model type")
    download_parser.add_argument("--filename", help="Filename for GGUF models")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model_id", help="Model ID")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("model_id", help="Model ID")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument("model_id", help="Model ID")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up model cache")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    if args.command == "download":
        if args.type == "huggingface":
            manager.download_huggingface_model(args.model_id, args.revision)
        elif args.type == "gguf":
            if not args.filename:
                print("Filename is required for GGUF models")
                return
            manager.download_gguf_model(args.model_id, args.filename)
    
    elif args.command == "list":
        models = manager.list_models()
        if not models:
            print("No models found")
            return
        
        print(f"{'Model ID':<40} {'Source':<15} {'Size':<10} {'Downloaded':<20}")
        print("-" * 90)
        for model in models:
            size = manager.format_size(model.get("size", 0))
            downloaded = model.get("downloaded_at", "Unknown")[:19]
            print(f"{model['model_id']:<40} {model['source']:<15} {size:<10} {downloaded:<20}")
    
    elif args.command == "info":
        info = manager.get_model_info(args.model_id)
        if not info:
            print(f"Model {args.model_id} not found")
            return
        
        print(json.dumps(info, indent=2))
    
    elif args.command == "delete":
        manager.delete_model(args.model_id)
    
    elif args.command == "verify":
        manager.verify_model(args.model_id)
    
    elif args.command == "cleanup":
        manager.cleanup_cache()

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/model-manager.py

# Create dataset management script
cat > ~/ai-workspace/tools/dataset-manager.py << 'EOF'
#!/usr/bin/env python3
"""
Dataset Management CLI for Omarchy AI
Provides tools for downloading, processing, and managing datasets
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm

class DatasetManager:
    def __init__(self, config_path: str = "~/.config/model-manager.yaml"):
        self.config_path = Path(config_path).expanduser()
        self.config = self.load_config()
        self.datasets_path = Path(self.config["storage"]["backends"][0]["path"]).parent / "datasets"
        self.datasets_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            sys.exit(1)
    
    def download_huggingface_dataset(self, dataset_id: str, split: str = None) -> str:
        """Download dataset from Hugging Face Hub."""
        print(f"Downloading dataset {dataset_id} from Hugging Face...")
        
        try:
            dataset = load_dataset(dataset_id, split=split)
            dataset_path = self.datasets_path / "huggingface" / dataset_id
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Save dataset
            if isinstance(dataset, Dataset):
                dataset.save_to_disk(str(dataset_path))
            else:
                for split_name, split_data in dataset.items():
                    split_path = dataset_path / split_name
                    split_data.save_to_disk(str(split_path))
            
            print(f"Dataset downloaded to: {dataset_path}")
            return str(dataset_path)
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def create_train_test_split(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        """Create train/test splits for a dataset."""
        print(f"Creating train/test split for {dataset_path}...")
        
        try:
            # Load dataset
            if Path(dataset_path).suffix == '.csv':
                df = pd.read_csv(dataset_path)
                
                # Split data
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=random_state
                )
                
                # Save splits
                base_path = Path(dataset_path).parent
                train_df.to_csv(base_path / "train.csv", index=False)
                test_df.to_csv(base_path / "test.csv", index=False)
                
                print(f"Train split: {len(train_df)} samples")
                print(f"Test split: {len(test_df)} samples")
                
            else:
                # For Hugging Face datasets
                dataset = Dataset.load_from_disk(dataset_path)
                train_test = dataset.train_test_split(test_size=test_size, seed=random_state)
                
                # Save splits
                base_path = Path(dataset_path).parent
                train_test["train"].save_to_disk(str(base_path / "train"))
                train_test["test"].save_to_disk(str(base_path / "test"))
                
                print(f"Train split: {len(train_test['train'])} samples")
                print(f"Test split: {len(train_test['test'])} samples")
                
        except Exception as e:
            print(f"Error creating splits: {e}")
    
    def list_datasets(self) -> List[str]:
        """List all downloaded datasets."""
        datasets = []
        for path in self.datasets_path.rglob("*"):
            if path.is_dir() and (path / "dataset_info.json").exists():
                datasets.append(str(path.relative_to(self.datasets_path)))
        return datasets
    
    def dataset_info(self, dataset_path: str) -> Dict:
        """Get dataset information."""
        full_path = self.datasets_path / dataset_path
        
        if not full_path.exists():
            return None
        
        info = {
            "path": str(full_path),
            "size": self.get_directory_size(str(full_path)),
            "files": list(full_path.glob("*"))
        }
        
        # Try to get dataset info from Hugging Face format
        info_file = full_path / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info.update(json.load(f))
        
        return info
    
    def get_directory_size(self, path: str) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    
    def format_size(self, size: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

def main():
    parser = argparse.ArgumentParser(description="AI Dataset Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., squad)")
    download_parser.add_argument("--split", help="Dataset split to download")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/test splits")
    split_parser.add_argument("dataset_path", help="Path to dataset")
    split_parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    split_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded datasets")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("dataset_path", help="Dataset path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DatasetManager()
    
    if args.command == "download":
        manager.download_huggingface_dataset(args.dataset_id, args.split)
    
    elif args.command == "split":
        manager.create_train_test_split(args.dataset_path, args.test_size, args.random_state)
    
    elif args.command == "list":
        datasets = manager.list_datasets()
        if not datasets:
            print("No datasets found")
            return
        
        print("Downloaded datasets:")
        for dataset in datasets:
            print(f"  {dataset}")
    
    elif args.command == "info":
        info = manager.dataset_info(args.dataset_path)
        if not info:
            print(f"Dataset {args.dataset_path} not found")
            return
        
        print(json.dumps(info, indent=2, default=str))

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/dataset-manager.py

# Create model serving script
cat > ~/ai-workspace/tools/model-server.py << 'EOF'
#!/usr/bin/env python3
"""
Model Serving Server for Omarchy AI
Provides REST API for model inference
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio
from contextlib import asynccontextmanager

# Global variables for model and tokenizer
model = None
tokenizer = None
pipe = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, tokenizer, pipe
    
    model_path = os.getenv("MODEL_PATH", "microsoft/DialoGPT-medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_path}")
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="~/ai-workspace/models/huggingface")
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="~/ai-workspace/models/huggingface")
        model.to(device)
        
        # Create pipeline for easier inference
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    print("Shutting down model server...")

app = FastAPI(
    title="Omarchy AI Model Server",
    description="REST API for AI model inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Omarchy AI Model Server", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Generate text
        outputs = pipe(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generation_time = time.time() - start_time
        generated_text = outputs[0]["generated_text"]
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": model.config.name_or_path,
        "model_type": model.config.model_type,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "device": next(model.parameters()).device.type,
        "parameters": sum(p.numel() for p in model.parameters())
    }

def main():
    parser = argparse.ArgumentParser(description="AI Model Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to model or model identifier")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    
    uvicorn.run(
        "model-server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/model-server.py

# Create systemd service for model server
sudo tee /etc/systemd/system/ai-model-server.service > /dev/null << 'EOF'
[Unit]
Description=AI Model Server
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=/home/${USER}/ai-workspace/tools
Environment=MODEL_PATH=microsoft/DialoGPT-medium
ExecStart=/usr/bin/python3 /home/${USER}/ai-workspace/tools/model-server.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Add model management aliases
cat >> ~/.bashrc << 'EOF'

# Model Management Aliases
alias model-download='python ~/ai-workspace/tools/model-manager.py download'
alias model-list='python ~/ai-workspace/tools/model-manager.py list'
alias model-info='python ~/ai-workspace/tools/model-manager.py info'
alias model-delete='python ~/ai-workspace/tools/model-manager.py delete'
alias model-verify='python ~/ai-workspace/tools/model-manager.py verify'
alias model-cleanup='python ~/ai-workspace/tools/model-manager.py cleanup'
alias model-serve='python ~/ai-workspace/tools/model-server.py'
alias model-server-start='sudo systemctl start ai-model-server'
alias model-server-stop='sudo systemctl stop ai-model-server'
alias model-server-status='sudo systemctl status ai-model-server'

# Dataset Management Aliases
alias dataset-download='python ~/ai-workspace/tools/dataset-manager.py download'
alias dataset-list='python ~/ai-workspace/tools/dataset-manager.py list'
alias dataset-info='python ~/ai-workspace/tools/dataset-manager.py info'
alias dataset-split='python ~/ai-workspace/tools/dataset-manager.py split'
EOF

echo "Model management system setup complete!"
echo "Available commands:"
echo "  model-download <model-id>  - Download model"
echo "  model-list                 - List downloaded models"
echo "  model-info <model-id>      - Show model information"
echo "  model-serve                - Start model server"
echo "  dataset-download <id>      - Download dataset"
echo "  dataset-list               - List datasets"