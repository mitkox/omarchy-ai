# Omarchy AI Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with your Omarchy AI installation.

## 🚨 Emergency Quick Fixes

### Installation Failed Completely
```bash
# Emergency recovery
~/.local/share/omarchy-ai/bin/omarchy-ai-repair --emergency

# Clean reinstall
rm -rf ~/.local/share/omarchy-ai
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash
```

### System Won't Boot After Installation
```bash
# Boot from live USB and chroot into your system
# Remove problematic display manager
sudo systemctl disable gdm
sudo systemctl disable lightdm
sudo systemctl disable sddm

# Reboot and use console to reinstall
```

### Conda Environment Broken
```bash
# Recreate ai-dev environment
conda env remove -n ai-dev -y
conda env create -f ~/.local/share/omarchy-ai/environment.yml
```

## 🔍 Diagnostic Commands

Run these commands to gather information about issues:

```bash
# System health check
omarchy-ai-doctor

# Quick functionality test  
omarchy-ai-test --quick

# System requirements validation
~/.local/share/omarchy-ai/validate-system.sh

# View installation logs
tail -50 ~/.omarchy-ai-install.log

# Check system resources
free -h && df -h && lscpu
```

## 📋 Common Issues & Solutions

### 1. Installation Issues

#### "System validation failed"
**Symptoms**: Installation stops with system requirement errors
**Solutions**:
```bash
# Check available RAM and disk space
free -h
df -h

# Free up disk space
sudo pacman -Sc  # Clear package cache
docker system prune -af  # Clear Docker cache (if installed)
rm -rf ~/.cache/*  # Clear user cache

# Increase swap if low on RAM
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### "yay not found"
**Symptoms**: Installation fails looking for AUR helper
**Solutions**:
```bash
# Install yay manually
cd /tmp
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si --noconfirm
```

#### "Conda installation failed"
**Symptoms**: Conda environment creation fails
**Solutions**:
```bash
# Manual conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Recreate environment
conda env create -f ~/.local/share/omarchy-ai/environment.yml
```

### 2. Environment Issues

#### "ai-env command not found"
**Symptoms**: Shell aliases not working
**Solutions**:
```bash
# Reload shell configuration
source ~/.bashrc

# Manually add aliases
cat >> ~/.bashrc << 'EOF'
alias ai-env='conda activate ai-dev'
alias ai-workspace='cd ~/ai-workspace'
alias jupyter-ai='cd ~/ai-workspace && jupyter lab'
EOF

source ~/.bashrc
```

#### "Cannot activate ai-dev environment"
**Symptoms**: Conda activation fails
**Solutions**:
```bash
# Check if environment exists
conda info --envs

# Recreate if missing
conda create -n ai-dev python=3.11 -y
conda activate ai-dev
pip install -r ~/.local/share/omarchy-ai/requirements.txt

# Fix conda initialization
conda init bash
source ~/.bashrc
```

### 3. GPU Issues

#### "NVIDIA GPU not detected"
**Symptoms**: nvidia-smi not working
**Solutions**:
```bash
# Check if GPU is physically detected
lspci | grep -i nvidia

# Install/reinstall NVIDIA drivers
sudo pacman -S nvidia nvidia-utils nvidia-settings

# For older GPUs
sudo pacman -S nvidia-470xx-dkms nvidia-470xx-utils

# Reboot after driver installation
sudo reboot
```

#### "CUDA not available in PyTorch"
**Symptoms**: torch.cuda.is_available() returns False
**Solutions**:
```bash
# Reinstall PyTorch with CUDA
conda activate ai-dev
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check CUDA version compatibility
nvidia-smi  # Note CUDA version
nvcc --version  # Should match or be compatible

# Install matching CUDA toolkit if needed
sudo pacman -S cuda-toolkit
```

#### "Out of memory errors on GPU"
**Symptoms**: CUDA out of memory during model loading
**Solutions**:
```bash
# Check GPU memory usage
nvidia-smi

# Set memory fraction in Python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use CPU fallback for large models
export CUDA_VISIBLE_DEVICES=""

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
```

### 4. Model Management Issues

#### "Cannot download models"
**Symptoms**: Model downloads fail or are slow
**Solutions**:
```bash
# Check network connectivity
curl -I https://huggingface.co

# Use mirror or proxy if needed
export HF_ENDPOINT=https://hf-mirror.com

# Download with git-lfs manually
cd ~/ai-workspace/models/huggingface
git lfs install
git clone https://huggingface.co/microsoft/DialoGPT-medium

# Increase timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

#### "Models taking too much disk space"
**Symptoms**: Running out of storage
**Solutions**:
```bash
# Clean model cache
model-cleanup

# Move models to external storage
sudo mkdir -p /mnt/external/ai-models
sudo chown $USER:$USER /mnt/external/ai-models
ln -sf /mnt/external/ai-models ~/ai-workspace/models

# Use symlinks for large models
mv ~/ai-workspace/models/large-model /mnt/external/
ln -s /mnt/external/large-model ~/ai-workspace/models/
```

### 5. Jupyter Issues

#### "Jupyter Lab won't start"
**Symptoms**: jupyter lab command fails
**Solutions**:
```bash
# Reinstall Jupyter
conda activate ai-dev
pip uninstall jupyter jupyterlab -y
pip install jupyter jupyterlab

# Reset Jupyter config
rm -rf ~/.jupyter
jupyter lab --generate-config

# Start with specific settings
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

#### "Kernel not found in Jupyter"
**Symptoms**: No Python kernel available
**Solutions**:
```bash
# Register ai-dev kernel
conda activate ai-dev
pip install ipykernel
python -m ipykernel install --user --name=ai-dev --display-name="AI Development"

# List available kernels
jupyter kernelspec list

# Remove broken kernels
jupyter kernelspec remove broken-kernel
```

### 6. Performance Issues

#### "System running very slowly"
**Symptoms**: High CPU/memory usage
**Solutions**:
```bash
# Check resource usage
htop
iotop
nvidia-smi

# Kill resource-heavy processes
pkill -f jupyter
pkill -f python

# Optimize system
# Increase swappiness for better memory management
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Disable unnecessary services
sudo systemctl disable cups
sudo systemctl disable bluetooth
```

#### "Model inference very slow"
**Symptoms**: Long response times
**Solutions**:
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Optimize CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use quantized models
model-download microsoft/DialoGPT-small  # Instead of medium/large

# Enable optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 7. Network Issues

#### "Cannot connect to package repositories"
**Symptoms**: Package downloads fail
**Solutions**:
```bash
# Update package databases
sudo pacman -Sy

# Check mirrorlist
sudo pacman -S reflector
sudo reflector --latest 20 --protocol https --sort rate --save /etc/pacman.d/mirrorlist

# Use different DNS servers
echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf
echo 'nameserver 8.8.4.4' | sudo tee -a /etc/resolv.conf
```

#### "Hugging Face downloads timing out"
**Symptoms**: Model downloads fail with timeouts
**Solutions**:
```bash
# Use HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# Increase timeouts
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Use resume functionality
HF_HUB_ENABLE_HF_TRANSFER=1 model-download model-name

# Download manually with git
cd ~/ai-workspace/models/huggingface
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/model-name
cd model-name
git lfs pull
```

## 🛠️ Advanced Troubleshooting

### System-Level Issues

#### Graphics/Display Issues
```bash
# Check display driver
lspci -k | grep -A 2 -E "(VGA|3D)"

# Reset Hyprland config
rm -rf ~/.config/hypr
~/.local/share/omarchy-ai/install/hyprlandia.sh

# Use software rendering temporarily
export LIBGL_ALWAYS_SOFTWARE=1
```

#### Audio Issues
```bash
# Check audio system
pulseaudio --check -v
systemctl --user restart pulseaudio

# Install missing audio packages
sudo pacman -S pulseaudio pavucontrol alsa-utils
```

#### Boot Issues
```bash
# Check systemd logs
journalctl -b -p err

# Disable problematic services temporarily
sudo systemctl disable problematic-service

# Boot to console mode
sudo systemctl set-default multi-user.target
```

### Application-Specific Issues

#### Docker Problems
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix permission issues
sudo chown $USER:$USER /var/run/docker.sock
```

#### Git/Version Control Issues
```bash
# Configure git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Fix git-lfs issues
git lfs install --system
git lfs pull
```

## 📊 Performance Optimization

### Memory Optimization
```bash
# Increase virtual memory limits
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf

# Optimize Python memory usage
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2
```

### GPU Optimization
```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=DISABLED
sudo nvidia-smi --auto-boost-permission=UNRESTRICTED

# Monitor GPU temperature
watch -n1 nvidia-smi
```

### Storage Optimization
```bash
# Enable SSD optimizations
echo 'deadline' | sudo tee /sys/block/sda/queue/scheduler

# Disable swap for SSDs (if enough RAM)
sudo swapoff -a
# Comment out swap in /etc/fstab
```

## 🆘 Getting Help

### Information to Collect
When seeking help, gather this information:
```bash
# System information
omarchy-ai-doctor > system-info.txt

# Hardware information
lscpu > hardware-info.txt
free -h >> hardware-info.txt
lspci >> hardware-info.txt

# Installation logs
cp ~/.omarchy-ai-install.log install-log.txt
cp ~/.omarchy-ai-test-results.json test-results.json
```

### Support Channels
1. **GitHub Issues**: https://github.com/mitkox/omarchy-ai/issues
2. **Documentation**: Check README.md and PRD.md
3. **Community**: Look for community forums or chat channels
4. **Self-Diagnosis**: Use `omarchy-ai-doctor` first

### Before Asking for Help
1. Run `omarchy-ai-doctor` and share the output
2. Try `omarchy-ai-repair` to fix common issues
3. Check if the issue is reproducible
4. Include your system specifications
5. Mention what you were trying to do when the issue occurred

## 🔄 Recovery Procedures

### Complete System Recovery
```bash
# Backup important data first
cp -r ~/ai-workspace ~/ai-workspace-backup

# Nuclear option - complete reinstall
rm -rf ~/.local/share/omarchy-ai
rm -rf ~/ai-workspace
rm ~/.omarchy-ai-*

# Fresh installation
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash

# Restore data
cp -r ~/ai-workspace-backup/projects ~/ai-workspace/
cp -r ~/ai-workspace-backup/notebooks ~/ai-workspace/
```

### Partial Recovery
```bash
# Repair specific components
omarchy-ai-repair

# Reinstall just AI components
source ~/.local/share/omarchy-ai/install/ai-development.sh
source ~/.local/share/omarchy-ai/install/model-management.sh
```

---

## 💡 Prevention Tips

1. **Regular Maintenance**:
   - Run `omarchy-ai-doctor` weekly
   - Update system regularly: `sudo pacman -Syu`
   - Clean cache monthly: `model-cleanup`

2. **Backup Strategy**:
   - Backup AI workspace: `tar -czf ai-workspace-backup.tar.gz ~/ai-workspace`
   - Export conda environment: `conda env export -n ai-dev > environment-backup.yml`

3. **Monitoring**:
   - Monitor disk space: `df -h`
   - Watch GPU usage: `nvidia-smi`
   - Check system load: `htop`

4. **Resource Management**:
   - Don't download too many large models at once
   - Clean temporary files regularly
   - Monitor memory usage during training

Remember: Most issues can be resolved by running the diagnostic and repair tools first!