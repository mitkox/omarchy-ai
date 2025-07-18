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

# Ensure git is available
if ! command -v git >/dev/null 2>&1; then
    echo "📦 Installing git..."
    sudo pacman -Sy --noconfirm --needed git
fi

echo -e "\n🚀 Cloning Omarchy AI..."
rm -rf ~/.local/share/omarchy-ai/

# Clone with progress and handle network issues
if ! git clone --progress https://github.com/mitkox/omarchy-ai.git ~/.local/share/omarchy-ai; then
    echo "❌ Failed to clone repository. Please check your internet connection."
    exit 1
fi

# Use custom branch if instructed
if [[ -n "$OMARCHY_AI_REF" ]]; then
  echo -e "\n🔄 Using branch: $OMARCHY_AI_REF"
  cd ~/.local/share/omarchy-ai
  if git fetch origin "${OMARCHY_AI_REF}" && git checkout "${OMARCHY_AI_REF}"; then
    echo "✅ Switched to branch $OMARCHY_AI_REF"
  else
    echo "⚠️  Failed to switch to branch $OMARCHY_AI_REF, using default branch"
  fi
  cd - >/dev/null
fi

# Ensure the repository is in the correct location for install script
echo -e "\n⚙️  Preparing installation environment..."

# Make sure we're in the right directory
cd ~/.local/share/omarchy-ai

# Make scripts executable
chmod +x install.sh 2>/dev/null || true
chmod +x boot.sh 2>/dev/null || true

echo -e "\n🎯 Starting AI development environment installation..."
echo -e "This will install:"
echo -e "  • Base Omarchy system (Hyprland, Waybar, etc.)"
echo -e "  • AI development tools (Python, PyTorch, etc.)"
echo -e "  • Conda environment for AI development"
echo -e "  • Jupyter Lab and ML libraries"
echo -e "  • GPU support and monitoring tools"
echo -e "\n⏱️  Installation will take 10-30 minutes depending on your system..."

# Run the installation
if source ~/.local/share/omarchy-ai/install.sh; then
    echo -e "\n🎉 Installation completed successfully!"
else
    echo -e "\n❌ Installation failed. Check the output above for errors."
    echo -e "💡 You can retry by running: source ~/.local/share/omarchy-ai/install.sh"
    exit 1
fi
