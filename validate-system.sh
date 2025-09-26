#!/bin/bash
# Omarchy AI System Validation Script
# Validates system requirements before installation

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="$HOME/.omarchy-ai-validation.log"
readonly MIN_RAM_GB=16
readonly MIN_DISK_GB=100
readonly RECOMMENDED_RAM_GB=32
readonly RECOMMENDED_DISK_GB=500

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" >> "$LOG_FILE"
    echo -e "$*"
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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        return 1
    fi
}

# Check distribution
check_distribution() {
    info "Checking Linux distribution..."
    
    if [[ ! -f /etc/arch-release ]]; then
        error "This script is designed for Arch Linux"
        echo "Current distribution: $(lsb_release -d 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME || echo 'Unknown')"
        return 1
    fi
    
    success "Arch Linux detected"
    return 0
}

# Check system resources
check_memory() {
    info "Checking system memory..."
    
    local total_ram_kb
    total_ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_ram_gb=$((total_ram_kb / 1024 / 1024))
    
    echo "Available RAM: ${total_ram_gb}GB"
    
    if [[ $total_ram_gb -lt $MIN_RAM_GB ]]; then
        error "Insufficient RAM: ${total_ram_gb}GB (minimum: ${MIN_RAM_GB}GB)"
        return 1
    elif [[ $total_ram_gb -lt $RECOMMENDED_RAM_GB ]]; then
        warning "RAM below recommended: ${total_ram_gb}GB (recommended: ${RECOMMENDED_RAM_GB}GB)"
        warning "AI development may be slower with limited RAM"
    else
        success "RAM check passed: ${total_ram_gb}GB"
    fi
    
    return 0
}

check_disk_space() {
    info "Checking disk space..."
    
    local available_space_kb
    available_space_kb=$(df / | tail -1 | awk '{print $4}')
    local available_space_gb=$((available_space_kb / 1024 / 1024))
    
    echo "Available disk space: ${available_space_gb}GB"
    
    if [[ $available_space_gb -lt $MIN_DISK_GB ]]; then
        error "Insufficient disk space: ${available_space_gb}GB (minimum: ${MIN_DISK_GB}GB)"
        return 1
    elif [[ $available_space_gb -lt $RECOMMENDED_DISK_GB ]]; then
        warning "Disk space below recommended: ${available_space_gb}GB (recommended: ${RECOMMENDED_DISK_GB}GB)"
        warning "You may need to manage model storage carefully"
    else
        success "Disk space check passed: ${available_space_gb}GB"
    fi
    
    return 0
}

# Check CPU
check_cpu() {
    info "Checking CPU..."
    
    local cpu_cores
    cpu_cores=$(nproc)
    local cpu_info
    cpu_info=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
    
    echo "CPU: $cpu_info"
    echo "CPU cores: $cpu_cores"
    
    if [[ $cpu_cores -lt 4 ]]; then
        warning "CPU has fewer than 4 cores: $cpu_cores"
        warning "AI training and inference may be slow"
    else
        success "CPU check passed: $cpu_cores cores"
    fi
    
    # Check for specific CPU features
    if grep -q avx2 /proc/cpuinfo; then
        success "AVX2 support detected (good for AI performance)"
    else
        warning "AVX2 not detected - AI performance may be reduced"
    fi
    
    return 0
}

# Check GPU
check_gpu() {
    info "Checking GPU..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
        
        if [[ -n "$gpu_info" ]]; then
            echo "NVIDIA GPU detected:"
            while IFS=, read -r gpu_name gpu_memory; do
                gpu_name=$(echo "$gpu_name" | xargs)
                gpu_memory=$(echo "$gpu_memory" | xargs)
                echo "  - $gpu_name (${gpu_memory}MB VRAM)"
                
                if [[ $gpu_memory -lt 4096 ]]; then
                    warning "GPU VRAM below 4GB - large models may not fit"
                elif [[ $gpu_memory -lt 8192 ]]; then
                    warning "GPU VRAM below 8GB - some large models may not fit"
                else
                    success "GPU VRAM sufficient: ${gpu_memory}MB"
                fi
            done <<< "$gpu_info"
            
            # Check CUDA version
            local cuda_version
            cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "")
            if [[ -n "$cuda_version" ]]; then
                echo "CUDA Version: $cuda_version"
                success "NVIDIA GPU with CUDA support detected"
            fi
        else
            warning "NVIDIA GPU detected but nvidia-smi failed"
        fi
    else
        warning "No NVIDIA GPU detected - will use CPU for AI inference"
        warning "Performance will be significantly slower without GPU acceleration"
    fi
    
    # Check for other GPUs
    if lspci | grep -i "vga\|3d\|display" | grep -v -i nvidia >/dev/null; then
        local other_gpu
        other_gpu=$(lspci | grep -i "vga\|3d\|display" | grep -v -i nvidia | head -1)
        echo "Other GPU detected: $other_gpu"
        warning "Non-NVIDIA GPU detected - AI acceleration not supported"
    fi
    
    return 0
}

# Check network connectivity
check_network() {
    info "Checking network connectivity..."
    
    local test_urls=(
        "github.com"
        "pypi.org"
        "huggingface.co"
        "raw.githubusercontent.com"
    )
    
    for url in "${test_urls[@]}"; do
        if curl -s --connect-timeout 5 "$url" >/dev/null 2>&1; then
            success "Connection to $url: OK"
        else
            warning "Connection to $url: FAILED"
            warning "This may cause issues during installation"
        fi
    done
    
    return 0
}

# Check required system packages
check_system_packages() {
    info "Checking required system packages..."
    
    local required_packages=(
        "git"
        "curl"
        "wget"
        "base-devel"
        "python"
        "python-pip"
    )
    
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if pacman -Qi "$package" >/dev/null 2>&1; then
            success "Package $package: installed"
        else
            missing_packages+=("$package")
            warning "Package $package: not installed"
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        warning "Missing packages detected. Install with:"
        echo "sudo pacman -S ${missing_packages[*]}"
        return 1
    fi
    
    return 0
}

# Check Python version
check_python() {
    info "Checking Python version..."
    
    if command -v python3 >/dev/null 2>&1; then
        local python_version
        python_version=$(python3 --version | awk '{print $2}')
        echo "Python version: $python_version"
        
        # Check if version is >= 3.9
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            success "Python version check passed"
        else
            error "Python version must be 3.9 or higher (found: $python_version)"
            return 1
        fi
    else
        error "Python 3 not found"
        return 1
    fi
    
    return 0
}

# Check for yay (AUR helper)
check_yay() {
    info "Checking for yay (AUR helper)..."
    
    if command -v yay >/dev/null 2>&1; then
        success "yay found"
    else
        warning "yay not found - required for AUR packages"
        warning "Install yay first or use the installation script to install it"
        return 1
    fi
    
    return 0
}

# Performance benchmarks
run_performance_benchmark() {
    info "Running basic performance benchmarks..."
    
    # CPU benchmark
    echo "Running CPU benchmark..."
    local cpu_benchmark_start
    cpu_benchmark_start=$(date +%s.%N)
    python3 -c "
import time
start = time.time()
# Simple CPU-intensive task
sum(i*i for i in range(100000))
end = time.time()
print(f'CPU benchmark: {end-start:.3f}s')
" 2>/dev/null || echo "CPU benchmark failed"
    
    # Memory benchmark
    echo "Running memory benchmark..."
    python3 -c "
import time
start = time.time()
# Allocate and access memory
data = [0] * 10000000
for i in range(len(data)):
    data[i] = i
end = time.time()
print(f'Memory benchmark: {end-start:.3f}s')
del data
" 2>/dev/null || echo "Memory benchmark failed"
    
    # Disk I/O benchmark
    echo "Running disk I/O benchmark..."
    local test_file="/tmp/omarchy_disk_test_$$"
    dd if=/dev/zero of="$test_file" bs=1M count=100 2>/dev/null && \
    sync && \
    dd if="$test_file" of=/dev/null bs=1M 2>/dev/null && \
    rm -f "$test_file" && \
    success "Disk I/O benchmark completed" || \
    warning "Disk I/O benchmark failed"
    
    return 0
}

# Generate system report
generate_system_report() {
    info "Generating system report..."
    
    local report_file="$HOME/.omarchy-ai-system-report.txt"
    
    {
        echo "Omarchy AI System Report"
        echo "========================"
        echo "Generated: $(date)"
        echo ""
        
        echo "System Information:"
        echo "- OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
        echo "- Kernel: $(uname -r)"
        echo "- Architecture: $(uname -m)"
        echo ""
        
        echo "Hardware:"
        echo "- CPU: $(lscpu | grep "Model name" | cut -d':' -f2 | xargs)"
        echo "- Cores: $(nproc)"
        echo "- RAM: $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))GB"
        echo "- Disk: $(($(df / | tail -1 | awk '{print $4}') / 1024 / 1024))GB available"
        echo ""
        
        echo "GPU Information:"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU detected but nvidia-smi failed"
        else
            echo "No NVIDIA GPU detected"
        fi
        echo ""
        
        echo "Python Environment:"
        python3 --version 2>/dev/null || echo "Python not available"
        pip3 --version 2>/dev/null || echo "pip not available"
        echo ""
        
        echo "Network Connectivity:"
        for url in github.com pypi.org huggingface.co; do
            if curl -s --connect-timeout 5 "$url" >/dev/null 2>&1; then
                echo "- $url: OK"
            else
                echo "- $url: FAILED"
            fi
        done
    } > "$report_file"
    
    success "System report generated: $report_file"
    return 0
}

# Main validation function
main() {
    local validation_passed=true
    local warnings_count=0
    
    info "Starting Omarchy AI system validation..."
    info "Log file: $LOG_FILE"
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Omarchy AI System Validation - $(date)" > "$LOG_FILE"
    
    # Run validation checks
    local checks=(
        "check_root"
        "check_distribution" 
        "check_memory"
        "check_disk_space"
        "check_cpu"
        "check_gpu"
        "check_python"
        "check_system_packages"
        "check_yay"
        "check_network"
    )
    
    for check in "${checks[@]}"; do
        if ! $check; then
            validation_passed=false
        fi
    done
    
    # Run benchmarks if requested
    if [[ "${1:-}" == "--benchmark" ]]; then
        run_performance_benchmark
    fi
    
    # Generate system report
    generate_system_report
    
    echo ""
    echo "================================"
    if [[ $validation_passed == true ]]; then
        success "System validation PASSED"
        success "Your system meets the minimum requirements for Omarchy AI"
        echo ""
        info "Next steps:"
        echo "1. Run the installation: ./install.sh"
        echo "2. Review the system report: cat ~/.omarchy-ai-system-report.txt"
    else
        error "System validation FAILED"
        error "Please address the issues above before installation"
        echo ""
        info "Common solutions:"
        echo "1. Install missing packages: sudo pacman -S <package-names>"
        echo "2. Install yay: git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si"
        echo "3. Free up disk space or add more RAM if needed"
        echo "4. Check the full log: cat $LOG_FILE"
        return 1
    fi
    
    return 0
}

# Help function
show_help() {
    cat << EOF
Omarchy AI System Validation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --help         Show this help message
    --benchmark    Run performance benchmarks
    --report-only  Generate system report only (skip validation)

EXAMPLES:
    $0                  # Run full validation
    $0 --benchmark      # Run validation with benchmarks
    $0 --report-only    # Generate system report only

FILES:
    ~/.omarchy-ai-validation.log      # Validation log
    ~/.omarchy-ai-system-report.txt   # System information report

REQUIREMENTS:
    - Arch Linux
    - Minimum 16GB RAM (32GB recommended)
    - Minimum 100GB disk space (500GB recommended) 
    - Python 3.9+
    - Internet connection for package downloads

EOF
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --benchmark)
        main --benchmark
        ;;
    --report-only)
        generate_system_report
        ;;
    *)
        main "$@"
        ;;
esac