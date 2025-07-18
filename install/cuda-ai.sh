# Enhanced CUDA Setup for AI Development
# Extends the existing nvidia.sh with AI-specific CUDA configurations

# Source the base nvidia configuration
source ~/.local/share/omarchy/install/nvidia.sh

# Install additional CUDA tools for AI development
if [ -n "$(lspci | grep -i 'nvidia')" ]; then
  echo "Installing additional CUDA tools for AI development..."
  
  yay -S --noconfirm --needed \
    cuda-toolkit \
    cudnn \
    nccl \
    tensorrt \
    python-pycuda \
    python-cupy
  
  # Install nvidia-ml-py for GPU monitoring
  pip install nvidia-ml-py3 gpustat
  
  # Create CUDA environment setup
  cat >> ~/.bashrc << 'EOF'

# CUDA Environment for AI Development
export CUDA_HOME=/opt/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export CUDACXX=/opt/cuda/bin/nvcc

# AI-specific CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
EOF

  # Create GPU monitoring script
  cat > ~/ai-workspace/tools/gpu-monitor.sh << 'EOF'
#!/bin/bash
# GPU Monitoring Script for AI Development

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo -e "\n=== GPU Processes ==="
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits

echo -e "\n=== CUDA Version ==="
nvcc --version | grep "release"

echo -e "\n=== Real-time GPU Usage ==="
echo "Press Ctrl+C to stop monitoring"
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk '"'"'BEGIN{FS=","} {printf "GPU: %s%% | Memory: %sMB/%sMB | Temp: %s°C\n", $1, $2, $3, $4}'"'"''
EOF

  chmod +x ~/ai-workspace/tools/gpu-monitor.sh

  # Create CUDA test script
  cat > ~/ai-workspace/tools/cuda-test.py << 'EOF'
#!/usr/bin/env python3
"""
CUDA Test Script for AI Development Environment
Tests PyTorch and TensorFlow CUDA functionality
"""

import sys
import torch
import tensorflow as tf
import numpy as np

def test_pytorch_cuda():
    """Test PyTorch CUDA functionality"""
    print("=== PyTorch CUDA Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
        
        # Test tensor operations
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"Matrix multiplication test: {z.shape} on {z.device}")
    
    print()

def test_tensorflow_cuda():
    """Test TensorFlow CUDA functionality"""
    print("=== TensorFlow CUDA Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    for gpu in gpus:
        print(f"GPU: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    if gpus:
        # Test tensor operations
        with tf.device('/GPU:0'):
            x = tf.random.normal((1000, 1000))
            y = tf.random.normal((1000, 1000))
            z = tf.matmul(x, y)
            print(f"Matrix multiplication test: {z.shape} on {z.device}")
    
    print()

def test_gpu_memory():
    """Test GPU memory allocation"""
    print("=== GPU Memory Test ===")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # Check initial memory
        print(f"Initial memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB")
        print(f"Initial memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f}MB")
        
        # Allocate some memory
        x = torch.randn(5000, 5000, device=device)
        
        print(f"After allocation: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB")
        print(f"After allocation cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f}MB")
        
        # Clean up
        del x
        torch.cuda.empty_cache()
        
        print(f"After cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB")
        print(f"After cleanup cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f}MB")

if __name__ == "__main__":
    print("CUDA AI Development Environment Test")
    print("=" * 40)
    
    test_pytorch_cuda()
    test_tensorflow_cuda()
    test_gpu_memory()
    
    print("Test completed!")
EOF

  chmod +x ~/ai-workspace/tools/cuda-test.py

  # Create GPU benchmarking script
  cat > ~/ai-workspace/tools/gpu-benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Benchmark Script for AI Development
Benchmarks GPU performance for common AI operations
"""

import time
import torch
import torch.nn as nn
import numpy as np

def benchmark_matrix_multiplication(size=5000, iterations=10):
    """Benchmark matrix multiplication"""
    print(f"Matrix Multiplication Benchmark ({size}x{size}, {iterations} iterations)")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU benchmark
    if torch.cuda.is_available():
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(x, y)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            z = torch.matmul(x, y)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.3f}s ({gpu_time/iterations:.3f}s per iteration)")
        
        # CPU benchmark for comparison
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        for _ in range(min(iterations, 3)):  # Fewer iterations for CPU
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.3f}s ({cpu_time/min(iterations, 3):.3f}s per iteration)")
        print(f"GPU speedup: {(cpu_time/min(iterations, 3))/(gpu_time/iterations):.1f}x")
    
    print()

def benchmark_neural_network(batch_size=1000, iterations=100):
    """Benchmark neural network training"""
    print(f"Neural Network Benchmark (batch_size={batch_size}, {iterations} iterations)")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Generate random data
    x = torch.randn(batch_size, 784, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # Warmup
    for _ in range(3):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(iterations):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    print(f"Training time: {total_time:.3f}s ({total_time/iterations:.3f}s per iteration)")
    print(f"Throughput: {batch_size * iterations / total_time:.0f} samples/second")
    print()

if __name__ == "__main__":
    print("GPU Benchmark for AI Development")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available, running CPU benchmarks only")
    
    benchmark_matrix_multiplication()
    benchmark_neural_network()
    
    print("Benchmark completed!")
EOF

  chmod +x ~/ai-workspace/tools/gpu-benchmark.py

  # Add GPU monitoring aliases
  cat >> ~/.bashrc << 'EOF'

# GPU Monitoring Aliases
alias gpu-monitor='~/ai-workspace/tools/gpu-monitor.sh'
alias gpu-test='python ~/ai-workspace/tools/cuda-test.py'
alias gpu-benchmark='python ~/ai-workspace/tools/gpu-benchmark.py'
alias gpu-stats='watch -n 1 nvidia-smi'
alias gpu-processes='nvidia-smi pmon'
EOF

  echo "Enhanced CUDA setup for AI development complete!"
  echo "Available commands:"
  echo "  gpu-monitor    - Monitor GPU usage"
  echo "  gpu-test       - Test CUDA functionality"
  echo "  gpu-benchmark  - Benchmark GPU performance"
  echo "  gpu-stats      - Real-time GPU statistics"
fi