ascii_art=' ▄██████▄    ▄▄▄▄███▄▄▄▄      ▄████████    ▄████████  ▄████████    ▄█    █▄    ▄██   ▄     ▄████████  ▄█  
███    ███ ▄██▀▀▀███▀▀▀██▄   ███    ███   ███    ███ ███    ███   ███    ███   ███   ██▄  ███    ███ ███  
███    ███ ███   ███   ███   ███    ███   ███    ███ ███    █▀    ███    ███   ███▄▄▄███  ███    ███ ███▌ 
███    ███ ███   ███   ███   ███    ███  ▄███▄▄▄▄██▀ ███         ▄███▄▄▄▄███▄▄ ▀▀▀▀▀▀███  ███    ███ ███▌ 
███    ███ ███   ███   ███ ▀███████████ ▀▀███▀▀▀▀▀   ███        ▀▀███▀▀▀▀███▀  ▄██   ███ ▀███████████ ███▌ 
███    ███ ███   ███   ███   ███    ███ ▀███████████ ███    █▄    ███    ███   ███   ███   ███    ███ ███  
███    ███ ███   ███   ███   ███    ███   ███    ███ ███    ███   ███    ███   ███   ███   ███    ███ ███  
 ▀██████▀   ▀█   ███   █▀    ███    █▀    ███    ███ ████████▀    ███    █▀     ▀█████▀    ███    █▀  █▀   
                                          ███    ███                                                        '

echo -e "\n$ascii_art\n"

# Enhanced error handling and logging
set -euo pipefail
readonly LOG_FILE="$HOME/.omarchy-ai-boot.log"
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}❌ ERROR: $*${NC}"
}

success() {
    log "${GREEN}✅ SUCCESS: $*${NC}"
}

info() {
    log "${BLUE}ℹ️  INFO: $*${NC}"
}

# Initialize log
echo "Omarchy AI Boot Script - $(date)" > "$LOG_FILE"

# Ensure git is available
if ! command -v git >/dev/null 2>&1; then
    info "Installing git..."
    if ! sudo pacman -Sy --noconfirm --needed git; then
        error "Failed to install git. Please install it manually before continuing."
        exit 1
    fi
    success "Git installed successfully"
fi

# Basic system validation
info "Running basic system validation..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "This script should not be run as root"
    exit 1
fi

# Check distribution
if [[ ! -f /etc/arch-release ]]; then
    error "This installer is designed for Arch Linux only"
    echo "Detected: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2 2>/dev/null || echo 'Unknown distribution')"
    exit 1
fi

# Check available space
available_space_gb=$(($(df / | tail -1 | awk '{print $4}') / 1024 / 1024))
if [[ $available_space_gb -lt 50 ]]; then
    error "Insufficient disk space: ${available_space_gb}GB available (minimum 50GB required for basic installation)"
    exit 1
fi

# Check RAM
total_ram_gb=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
if [[ $total_ram_gb -lt 8 ]]; then
    error "Insufficient RAM: ${total_ram_gb}GB (minimum 8GB required, 16GB recommended)"
    exit 1
elif [[ $total_ram_gb -lt 16 ]]; then
    log "${YELLOW}⚠️  WARNING: RAM below recommended: ${total_ram_gb}GB (16GB+ recommended for optimal performance)${NC}"
fi

info "System validation passed: ${total_ram_gb}GB RAM, ${available_space_gb}GB available space"

info "Cloning Omarchy AI repository..."
rm -rf ~/.local/share/omarchy-ai/

# Clone with enhanced error handling and progress
if ! git clone --progress https://github.com/mitkox/omarchy-ai.git ~/.local/share/omarchy-ai; then
    error "Failed to clone repository"
    echo "This could be due to:"
    echo "  • No internet connection"
    echo "  • GitHub is unreachable"
    echo "  • Insufficient disk space"
    echo ""
    echo "Please check your network connection and try again."
    exit 1
fi

success "Repository cloned successfully"

# Use custom branch if instructed
if [[ -n "${OMARCHY_AI_REF:-}" ]]; then
    info "Switching to branch: $OMARCHY_AI_REF"
    cd ~/.local/share/omarchy-ai
    if git fetch origin "${OMARCHY_AI_REF}" && git checkout "${OMARCHY_AI_REF}"; then
        success "Switched to branch $OMARCHY_AI_REF"
    else
        log "${YELLOW}⚠️  Failed to switch to branch $OMARCHY_AI_REF, using default branch${NC}"
    fi
    cd - >/dev/null
fi

# Prepare installation environment
info "Preparing installation environment..."

# Make sure we're in the right directory
cd ~/.local/share/omarchy-ai

# Make scripts executable
chmod +x install.sh 2>/dev/null || true
chmod +x boot.sh 2>/dev/null || true
chmod +x validate-system.sh 2>/dev/null || true
chmod +x bin/* 2>/dev/null || true

# Run system validation if available
if [[ -x validate-system.sh ]]; then
    info "Running comprehensive system validation..."
    if ./validate-system.sh; then
        success "System validation completed successfully"
    else
        error "System validation failed - some issues detected"
        echo ""
        echo "You can:"
        echo "  1. Check the validation report: cat ~/.omarchy-ai-system-report.txt"
        echo "  2. Continue installation anyway (not recommended)"
        echo "  3. Fix the issues and try again"
        echo ""
        
        read -p "Continue with installation despite validation warnings? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            info "Installation cancelled by user"
            exit 1
        fi
    fi
fi

info "Starting AI development environment installation..."
echo ""
echo "📋 This will install:"
echo "   • Base Omarchy system (Hyprland, Waybar, themes)"
echo "   • AI development tools (Python, PyTorch, TensorFlow)"
echo "   • Conda environment 'ai-dev' with ML libraries"
echo "   • Jupyter Lab and development tools"
echo "   • GPU support and monitoring (if NVIDIA GPU detected)"
echo "   • Model management system"
echo "   • Local inference server (llama.cpp)"
echo "   • Container support and CI/CD tools"
echo "   • Offline documentation and examples"
echo ""
echo "⏱️  Installation typically takes 15-45 minutes depending on:"
echo "   • Internet connection speed"
echo "   • System performance"
echo "   • Whether GPU support is installed"
echo ""

# Confirm installation
if [[ "${OMARCHY_AI_AUTO_CONFIRM:-}" != "true" ]]; then
    read -p "Proceed with installation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Installation cancelled by user"
        exit 0
    fi
fi

# Run the installation with enhanced error handling
info "Launching installation script..."
if source ~/.local/share/omarchy-ai/install.sh; then
    success "🎉 Omarchy AI installation completed successfully!"
    echo ""
    echo "🚀 What's next?"
    echo "   1. Reboot your system: sudo reboot"
    echo "   2. After reboot, test your installation: omarchy-ai-doctor"
    echo "   3. Start developing: ai-env && jupyter-ai"
    echo ""
    echo "📚 Resources:"
    echo "   • Quick Start: cat ~/.local/share/omarchy-ai/QUICKSTART.md"
    echo "   • Troubleshooting: cat ~/.local/share/omarchy-ai/TROUBLESHOOTING.md"
    echo "   • AI Workspace: ~/ai-workspace/"
    echo ""
    echo "💡 Run 'omarchy-ai-test' to verify everything works correctly"
else
    error "❌ Installation failed!"
    echo ""
    echo "🔍 Troubleshooting steps:"
    echo "   1. Check the installation log: tail -50 ~/.omarchy-ai-install.log"
    echo "   2. Run diagnostics: ~/.local/share/omarchy-ai/bin/omarchy-ai-doctor"
    echo "   3. Try repair: ~/.local/share/omarchy-ai/bin/omarchy-ai-repair"
    echo "   4. For complete reinstall: rm -rf ~/.local/share/omarchy-ai && curl -fsSL ... | bash"
    echo ""
    echo "💬 Get help:"
    echo "   • GitHub Issues: https://github.com/mitkox/omarchy-ai/issues"
    echo "   • Include logs and system info in your report"
    echo ""
    echo "📋 System info saved to: ~/.omarchy-ai-system-report.txt"
    exit 1
fi
