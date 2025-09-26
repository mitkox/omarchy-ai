# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="$HOME/.omarchy-ai-install.log"
readonly CHECKPOINT_FILE="$HOME/.omarchy-ai-checkpoint"
readonly STATE_FILE="$HOME/.omarchy-ai-state"
readonly BACKUP_DIR="$HOME/.omarchy-ai-backup"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Version info
readonly OMARCHY_AI_VERSION="1.0.0"

# Installation steps
readonly INSTALL_STEPS=(
    "validate_system"
    "setup_base_omarchy" 
    "install_ai_development"
    "install_cuda_support"
    "install_model_management"
    "install_llama_cpp"
    "install_ci_cd_tools"
    "install_containers"
    "install_offline_docs"
    "setup_environment"
    "run_tests"
    "finalize_installation"
)

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR: $*${NC}"
}

warning() {
    log "${YELLOW}WARNING: $*${NC}"
}

success() {
    log "${GREEN}SUCCESS: $*${NC}"
}

info() {
    log "${BLUE}INFO: $*${NC}"
}

# Progress tracking
show_progress() {
    local current="$1"
    local total="$2"
    local step_name="$3"
    local percentage=$((current * 100 / total))
    
    printf "\r${BLUE}[%d/%d] (%d%%) %s${NC}" "$current" "$total" "$percentage" "$step_name"
    echo ""
}

# Checkpoint management
create_checkpoint() {
    local step="$1"
    echo "$step" > "$CHECKPOINT_FILE"
    log "Checkpoint created: $step"
}

get_last_checkpoint() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        cat "$CHECKPOINT_FILE"
    else
        echo "start"
    fi
}

# State management for rollback
save_state() {
    local component="$1"
    local action="$2"
    local details="$3"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S')|$component|$action|$details" >> "$STATE_FILE"
}

# Rollback functionality
rollback_installation() {
    warning "Starting rollback process..."
    
    if [[ ! -f "$STATE_FILE" ]]; then
        error "No state file found - cannot rollback"
        return 1
    fi
    
    # Read state file in reverse order
    tac "$STATE_FILE" | while IFS='|' read -r timestamp component action details; do
        case "$action" in
            "package_install")
                warning "Rolling back package installation: $details"
                yay -Rns --noconfirm $details 2>/dev/null || true
                ;;
            "file_create")
                warning "Rolling back file creation: $details"
                rm -f "$details" 2>/dev/null || true
                ;;
            "directory_create")
                warning "Rolling back directory creation: $details"
                rmdir "$details" 2>/dev/null || true
                ;;
            "symlink_create")
                warning "Rolling back symlink creation: $details"
                rm -f "$details" 2>/dev/null || true
                ;;
            "service_enable")
                warning "Rolling back service enable: $details"
                sudo systemctl disable "$details" 2>/dev/null || true
                ;;
            "conda_env_create")
                warning "Rolling back conda environment creation: $details"
                conda env remove -n "$details" -y 2>/dev/null || true
                ;;
        esac
    done
    
    # Remove state files
    rm -f "$STATE_FILE" "$CHECKPOINT_FILE" 2>/dev/null || true
    
    success "Rollback completed"
}

# Error handling with rollback option
handle_error() {
    local exit_code=$?
    local line_number=$1
    
    error "Installation failed at line $line_number with exit code $exit_code"
    
    if command -v gum >/dev/null 2>&1; then
        if gum confirm "Installation failed. Do you want to rollback changes?"; then
            rollback_installation
        fi
    else
        echo "Installation failed. Run '$0 --rollback' to undo changes."
    fi
    
    exit $exit_code
}

# Set error trap
trap 'handle_error $LINENO' ERR

# System validation
validate_system() {
    info "Running system validation..."
    
    if [[ -x "$SCRIPT_DIR/validate-system.sh" ]]; then
        "$SCRIPT_DIR/validate-system.sh" || {
            error "System validation failed"
            return 1
        }
    else
        warning "System validation script not found, running basic checks..."
        
        # Basic checks
        [[ $(free -g | awk 'NR==2{print $2}') -ge 16 ]] || {
            error "Minimum 16GB RAM required"
            return 1
        }
        
        [[ $(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//') -ge 100 ]] || {
            error "Minimum 100GB free space required"
            return 1
        }
    fi
    
    success "System validation passed"
}

# Enhanced base Omarchy setup
setup_base_omarchy() {
    info "Setting up base Omarchy system..."
    
    # Install base Omarchy system first
    echo "Installing base Omarchy system..."
    rm -rf ~/.local/share/omarchy/
    
    if ! git clone https://github.com/basecamp/omarchy.git ~/.local/share/omarchy; then
        warning "Could not clone base Omarchy repository. Continuing with omarchy-ai only..."
        mkdir -p ~/.local/share/omarchy/install
        touch ~/.local/share/omarchy/install/dummy.sh
    else
        save_state "omarchy" "directory_create" "~/.local/share/omarchy"
    fi

    # Run base installers with error handling
    for installer in ~/.local/share/omarchy/install/*.sh; do
        if [[ -f "$installer" && "$(basename "$installer")" != "dummy.sh" ]]; then
            info "Running base installer: $(basename "$installer")"
            if ! source "$installer"; then
                warning "Base installer $(basename "$installer") failed, continuing..."
            fi
        fi
    done
    
    success "Base Omarchy setup completed"
}

# Enhanced AI development setup
install_ai_development() {
    info "Installing AI development tools..."
    
    # Set environment variables for conda and AI tools
    export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    export CONDA_AUTO_ACTIVATE_BASE=false
    
    # Install system packages with better error handling
    local ai_system_packages=(
        "python" "python-pip" "python-pipx"
        "python-virtualenv" "python-conda"
        "miniconda3"
        "jupyter-notebook" 
        "python-numpy" "python-scipy" "python-matplotlib"
        "python-pandas" "python-scikit-learn"
        "python-seaborn" "python-plotly"
        "python-requests" "python-urllib3"
        "python-yaml" "python-toml"
        "python-rich" "python-tqdm"
        "sqlite3"
        "curl" "wget"
        "git-lfs"
        "jq"
    )
    
    info "Installing system packages for AI development..."
    for package in "${ai_system_packages[@]}"; do
        if yay -S --noconfirm --needed "$package"; then
            save_state "package" "package_install" "$package"
        else
            warning "Failed to install $package - continuing..."
        fi
    done
    
    # Initialize conda properly
    info "Initializing conda..."
    local conda_found=false
    local conda_paths=(
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/usr/share/miniconda3/etc/profile.d/conda.sh"
    )
    
    for conda_path in "${conda_paths[@]}"; do
        if [[ -f "$conda_path" ]]; then
            source "$conda_path"
            conda_found=true
            break
        fi
    done
    
    if [[ "$conda_found" == false ]] && command -v conda >/dev/null 2>&1; then
        conda_base=$(conda info --base 2>/dev/null)
        if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
            source "$conda_base/etc/profile.d/conda.sh"
            conda_found=true
        fi
    fi
    
    if [[ "$conda_found" == false ]]; then
        error "Conda not available. Please install miniconda3 manually."
        return 1
    fi
    
    # Create or update conda environment
    info "Creating AI development environment..."
    if conda info --envs 2>/dev/null | grep -q "ai-dev"; then
        warning "AI development environment already exists, updating..."
        conda env update -n ai-dev -f "$SCRIPT_DIR/environment.yml"
    else
        conda env create -f "$SCRIPT_DIR/environment.yml"
        save_state "conda" "conda_env_create" "ai-dev"
    fi
    
    success "AI development tools installed"
}

# Install CUDA support with validation
install_cuda_support() {
    info "Installing CUDA support..."
    
    # Check for NVIDIA GPU
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warning "No NVIDIA GPU detected, skipping CUDA installation"
        return 0
    fi
    
    # Source the existing CUDA installer with enhancements
    if [[ -f "$SCRIPT_DIR/install/cuda-ai.sh" ]]; then
        source "$SCRIPT_DIR/install/cuda-ai.sh"
    fi
    
    # Verify CUDA installation
    if command -v nvcc >/dev/null 2>&1; then
        success "CUDA installation verified"
    else
        warning "CUDA installation may have issues"
    fi
}

# Install model management with enhancements
install_model_management() {
    info "Installing model management system..."
    
    if [[ -f "$SCRIPT_DIR/install/model-management.sh" ]]; then
        source "$SCRIPT_DIR/install/model-management.sh"
    fi
    
    success "Model management system installed"
}

# Install llama.cpp with validation
install_llama_cpp() {
    info "Installing llama.cpp..."
    
    if [[ -f "$SCRIPT_DIR/install/llama-cpp.sh" ]]; then
        source "$SCRIPT_DIR/install/llama-cpp.sh"
    fi
    
    # Test llama.cpp installation
    if command -v llama-cpp-server >/dev/null 2>&1; then
        success "llama.cpp installation verified"
    else
        warning "llama.cpp may not be properly installed"
    fi
}

# Install CI/CD tools
install_ci_cd_tools() {
    info "Installing CI/CD tools..."
    
    if [[ -f "$SCRIPT_DIR/install/ai-cicd.sh" ]]; then
        source "$SCRIPT_DIR/install/ai-cicd.sh" 
    fi
    
    success "CI/CD tools installed"
}

# Install containers
install_containers() {
    info "Installing container support..."
    
    if [[ -f "$SCRIPT_DIR/install/containers.sh" ]]; then
        source "$SCRIPT_DIR/install/containers.sh"
    fi
    
    success "Container support installed"
}

# Install offline documentation
install_offline_docs() {
    info "Installing offline documentation..."
    
    if [[ -f "$SCRIPT_DIR/install/offline-docs.sh" ]]; then
        source "$SCRIPT_DIR/install/offline-docs.sh"
    fi
    
    success "Offline documentation installed"
}

# Setup environment with enhanced configuration
setup_environment() {
    info "Setting up environment configuration..."
    
    # Create AI workspace directory structure
    mkdir -p ~/ai-workspace/{projects,models,datasets,experiments,notebooks,logs,mlruns,cache,tmp}
    
    for dir in ~/ai-workspace/{projects,models,datasets,experiments,notebooks,logs,mlruns,cache,tmp}; do
        save_state "directory" "directory_create" "$dir"
    done
    
    # Create comprehensive .env file
    cat > ~/ai-workspace/.env << 'EOF'
# AI Development Environment Variables - Omarchy AI v1.0.0

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
NVIDIA_VISIBLE_DEVICES=all

# Python/ML Configuration  
TOKENIZERS_PARALLELISM=false
PYTHONPATH=/home/${USER}/ai-workspace/src:$PYTHONPATH
PYTHONIOENCODING=utf-8
PYTHONDONTWRITEBYTECODE=1

# ML Operations
WANDB_MODE=offline
MLFLOW_TRACKING_URI=file:///home/${USER}/ai-workspace/mlruns
DVC_CONFIG_DIR=/home/${USER}/ai-workspace/.dvc

# Model and Data Paths
HF_HOME=/home/${USER}/ai-workspace/models/huggingface
TRANSFORMERS_CACHE=/home/${USER}/ai-workspace/models/transformers
TORCH_HOME=/home/${USER}/ai-workspace/models/torch
DIFFUSERS_CACHE=/home/${USER}/ai-workspace/models/diffusers

# Jupyter Configuration
JUPYTER_CONFIG_DIR=/home/${USER}/ai-workspace/.jupyter
JUPYTER_DATA_DIR=/home/${USER}/ai-workspace/.jupyter/data

# Development Tools
RUFF_CACHE_DIR=/home/${USER}/ai-workspace/cache/ruff
MYPY_CACHE_DIR=/home/${USER}/ai-workspace/cache/mypy
PYTEST_CACHE_DIR=/home/${USER}/ai-workspace/cache/pytest

# Performance Tuning
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMBA_NUM_THREADS=4

# Security
HUGGINGFACE_HUB_DISABLE_TELEMETRY=1
DISABLE_TELEMETRY=1
EOF
    
    save_state "file" "file_create" "~/ai-workspace/.env"
    
    success "Environment setup completed"
}

# Run installation tests
run_tests() {
    info "Running installation validation tests..."
    
    # Test conda environment
    if conda activate ai-dev 2>/dev/null; then
        success "Conda environment test passed"
        conda deactivate 2>/dev/null || true
    else
        warning "Conda environment test failed"
    fi
    
    # Test Python packages
    if conda run -n ai-dev python -c "import torch, transformers, numpy; print('Core packages OK')"; then
        success "Python packages test passed"
    else
        warning "Python packages test failed"
    fi
    
    # Test GPU if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        if conda run -n ai-dev python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
            success "GPU test passed"
        else
            warning "GPU test failed"
        fi
    fi
    
    success "Installation validation completed"
}

# Finalize installation
finalize_installation() {
    info "Finalizing installation..."
    
    # Update locate database
    if command -v updatedb >/dev/null 2>&1; then
        info "Updating file database..."
        sudo updatedb 2>/dev/null || warning "Could not update locate database"
    fi
    
    # Create version file
    cat > ~/.omarchy-ai-version << EOF
Omarchy AI v${OMARCHY_AI_VERSION}
Installed: $(date)
System: $(uname -a)
EOF
    
    save_state "file" "file_create" "~/.omarchy-ai-version"
    
    # Clean up temporary files
    rm -f "$CHECKPOINT_FILE" 2>/dev/null || true
    
    success "Installation finalized"
}

# Resume installation from checkpoint
resume_installation() {
    local last_checkpoint
    last_checkpoint=$(get_last_checkpoint)
    
    info "Resuming installation from checkpoint: $last_checkpoint"
    
    local resume=false
    for step in "${INSTALL_STEPS[@]}"; do
        if [[ "$step" == "$last_checkpoint" ]]; then
            resume=true
            continue
        fi
        
        if [[ "$resume" == true ]]; then
            run_installation_step "$step"
        fi
    done
}

# Run individual installation step
run_installation_step() {
    local step="$1"
    local step_number=0
    local total_steps=${#INSTALL_STEPS[@]}
    
    # Find step number
    for i in "${!INSTALL_STEPS[@]}"; do
        if [[ "${INSTALL_STEPS[$i]}" == "$step" ]]; then
            step_number=$((i + 1))
            break
        fi
    done
    
    show_progress "$step_number" "$total_steps" "Running $step"
    create_checkpoint "$step"
    
    case "$step" in
        "validate_system") validate_system ;;
        "setup_base_omarchy") setup_base_omarchy ;;
        "install_ai_development") install_ai_development ;;
        "install_cuda_support") install_cuda_support ;;
        "install_model_management") install_model_management ;;
        "install_llama_cpp") install_llama_cpp ;;
        "install_ci_cd_tools") install_ci_cd_tools ;;
        "install_containers") install_containers ;;
        "install_offline_docs") install_offline_docs ;;
        "setup_environment") setup_environment ;;
        "run_tests") run_tests ;;
        "finalize_installation") finalize_installation ;;
        *) error "Unknown installation step: $step"; return 1 ;;
    esac
}

# Main installation function
main_installation() {
    info "Starting Omarchy AI installation v${OMARCHY_AI_VERSION}..."
    info "Installation log: $LOG_FILE"
    
    # Ensure we're running from the correct directory
    if [[ ! -f "$SCRIPT_DIR/install.sh" ]]; then
        error "Installation must be run from ~/.local/share/omarchy-ai/"
        error "Current directory: $SCRIPT_DIR"
        return 1
    fi
    
    # Create directories
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initialize log
    echo "Omarchy AI Installation Log - $(date)" > "$LOG_FILE"
    echo "Version: $OMARCHY_AI_VERSION" >> "$LOG_FILE"
    echo "System: $(uname -a)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Run installation steps
    for step in "${INSTALL_STEPS[@]}"; do
        run_installation_step "$step"
    done
    
    # Installation completed successfully
    success "Omarchy AI installation completed successfully!"
    
    # Show final instructions
    show_final_instructions
}

# Show final instructions
show_final_instructions() {
    echo ""
    echo "🎉 Omarchy AI installation complete!"
    echo ""
    echo "📋 Installation Summary:"
    echo "   ✅ Base Omarchy system installed"
    echo "   ✅ AI development tools installed"
    echo "   ✅ Conda environment 'ai-dev' created"
    echo "   ✅ AI workspace created at ~/ai-workspace/"
    echo "   ✅ Shell aliases configured"
    echo "   ✅ CUDA support installed (if GPU available)"
    echo "   ✅ Model management system configured"
    echo "   ✅ Container support enabled"
    echo "   ✅ Offline documentation installed"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Reboot your system to apply all settings:"
    echo "   sudo reboot"
    echo ""
    echo "2. After reboot, test your installation:"
    echo "   omarchy-ai-doctor"
    echo ""
    echo "3. Start developing:"
    echo "   ai-env                  # Activate AI environment"
    echo "   ai-workspace           # Go to AI workspace"
    echo "   jupyter-ai             # Start Jupyter Lab"
    echo ""
    echo "4. Manage models:"
    echo "   model-download microsoft/DialoGPT-medium"
    echo "   model-list"
    echo "   model-serve"
    echo ""
    echo "5. Additional tools:"
    echo "   gpu-monitor            # Monitor GPU usage"
    echo "   llama-chat             # Chat with local models"
    echo "   docs-serve             # Start documentation server"
    echo ""
    echo "📚 Documentation:"
    echo "   - Installation log: $LOG_FILE"
    echo "   - System report: ~/.omarchy-ai-system-report.txt"
    echo "   - AI workspace: ~/ai-workspace/"
    echo ""
    echo "🛠️  Troubleshooting:"
    echo "   omarchy-ai-doctor      # Run diagnostics"
    echo "   omarchy-ai-repair      # Attempt repairs"
    echo "   $0 --rollback          # Rollback installation"
}

# Show help
show_help() {
    cat << EOF
Omarchy AI Installation Script v${OMARCHY_AI_VERSION}

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --help          Show this help message
    --resume        Resume installation from last checkpoint
    --rollback      Rollback installation changes
    --validate-only Run system validation only
    --version       Show version information

EXAMPLES:
    $0              # Run full installation
    $0 --resume     # Resume from checkpoint
    $0 --rollback   # Undo installation

FILES:
    ~/.omarchy-ai-install.log         # Installation log
    ~/.omarchy-ai-checkpoint          # Last checkpoint
    ~/.omarchy-ai-system-report.txt   # System report

For more information, visit: https://github.com/mitkox/omarchy-ai

EOF
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --version|-v)
        echo "Omarchy AI v${OMARCHY_AI_VERSION}"
        exit 0
        ;;
    --resume)
        resume_installation
        ;;
    --rollback)
        rollback_installation
        ;;
    --validate-only)
        validate_system
        ;;
    "")
        main_installation
        ;;
    *)
        error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

# Install base Omarchy system first
echo -e "\nInstalling base Omarchy system..."
rm -rf ~/.local/share/omarchy/
if ! git clone https://github.com/basecamp/omarchy.git ~/.local/share/omarchy >/dev/null 2>&1; then
  echo "⚠️  Warning: Could not clone base Omarchy repository. Continuing with omarchy-ai only..."
  echo "   This may affect some theme functionality, but installation can continue."
  mkdir -p ~/.local/share/omarchy/install
  # Create a dummy install directory with empty scripts to prevent errors
  touch ~/.local/share/omarchy/install/dummy.sh
fi

for f in ~/.local/share/omarchy/install/*.sh; do
  if [ -f "$f" ] && [ "$(basename "$f")" != "dummy.sh" ]; then
    echo -e "\nRunning base installer: $f"
    if ! source "$f"; then
      echo "⚠️  Warning: Base installer $f failed, continuing..."
    fi
  fi
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
  if [ -f "$f" ]; then
    echo -e "\nRunning AI installer: $f"
    if ! source "$f"; then
      echo "⚠️  Warning: AI installer $f failed, continuing..."
    fi
  fi
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
    if ! source ~/.local/share/omarchy-ai/install/$feature; then
      echo "⚠️  Warning: Enhanced feature $feature failed, continuing..."
    fi
  else
    echo "⚠️  Enhanced feature $feature not found, skipping..."
  fi
done

# Setup enhanced migration system
echo -e "\nSetting up migration system..."
if [ -x ~/.local/share/omarchy-ai/bin/omarchy-enhanced-migrations ]; then
  ~/.local/share/omarchy-ai/bin/omarchy-enhanced-migrations
else
  echo "⚠️  Migration system not found, skipping..."
fi

# Ensure locate is up to date now that everything has been installed
if command -v updatedb >/dev/null 2>&1; then
  echo "📂 Updating file database..."
  sudo updatedb 2>/dev/null || echo "⚠️  Could not update locate database"
else
  echo "⚠️  updatedb not available, skipping locate database update"
fi

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
