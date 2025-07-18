#!/bin/bash

# AI Environment Activation Script
# This script properly activates the AI development environment

echo "🚀 Activating AI Development Environment..."

# Enable hashing if disabled
set +h 2>/dev/null || true

# Set OpenSSL workaround
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

# Initialize conda if available
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    echo "✅ Conda initialized"
else
    echo "❌ Conda not found at /opt/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

# Try to activate the environment
if conda activate ai-dev 2>/dev/null; then
    echo "✅ AI development environment activated!"
    echo "📦 Python version: $(python --version)"
    echo "🌟 Environment: ai-dev"
    echo ""
    echo "Available commands:"
    echo "  python    - Python interpreter"
    echo "  pip       - Package installer"
    echo "  jupyter   - Jupyter notebook/lab"
    echo "  conda     - Conda package manager"
    echo ""
    echo "To deactivate: conda deactivate"
    
    # Start a new shell with the environment activated
    exec bash
else
    echo "❌ Failed to activate ai-dev environment"
    echo "💡 Try running this in a new terminal window"
    exit 1
fi
