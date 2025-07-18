# llama.cpp Installation and Setup for Omarchy AI
# This script installs and configures llama.cpp for local AI inference

# Install dependencies for building llama.cpp
yay -S --noconfirm --needed \
  cmake make gcc \
  cuda cuda-tools \
  openblas \
  pkg-config

# Clone and build llama.cpp with CUDA support
echo "Cloning and building llama.cpp with CUDA support..."
mkdir -p ~/ai-workspace/tools
cd ~/ai-workspace/tools

# Remove existing installation if present
rm -rf llama.cpp

# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support
make clean
make LLAMA_CUBLAS=1 -j$(nproc)

# Install llama.cpp binaries to system
sudo mkdir -p /usr/local/bin
sudo cp main /usr/local/bin/llama-main
sudo cp server /usr/local/bin/llama-server
sudo cp quantize /usr/local/bin/llama-quantize
sudo cp perplexity /usr/local/bin/llama-perplexity

# Make binaries executable
sudo chmod +x /usr/local/bin/llama-*

# Create symlinks for easier access
sudo ln -sf /usr/local/bin/llama-main /usr/local/bin/llama
sudo ln -sf /usr/local/bin/llama-server /usr/local/bin/llama-serve

# Create models directory
mkdir -p ~/ai-workspace/models/{llama,gguf}

# Create llama.cpp configuration
mkdir -p ~/.config/llama-cpp
cat > ~/.config/llama-cpp/config.json << 'EOF'
{
  "model_path": "/home/${USER}/ai-workspace/models/gguf",
  "default_model": "llama-2-7b-chat.Q4_K_M.gguf",
  "server_config": {
    "host": "127.0.0.1",
    "port": 8080,
    "threads": 8,
    "ctx_size": 4096,
    "n_gpu_layers": 35
  },
  "inference_config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "seed": -1
  }
}
EOF

# Create llama.cpp service file for systemd
sudo tee /etc/systemd/system/llama-server.service > /dev/null << 'EOF'
[Unit]
Description=Llama.cpp Inference Server
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=/home/${USER}/ai-workspace
ExecStart=/usr/local/bin/llama-server \
  --model /home/${USER}/ai-workspace/models/gguf/current-model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --threads 8 \
  --ctx-size 4096 \
  --n-gpu-layers 35
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create model management scripts
cat > ~/ai-workspace/tools/download-model.sh << 'EOF'
#!/bin/bash
# Download and setup GGUF models for llama.cpp

MODEL_DIR="$HOME/ai-workspace/models/gguf"
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

download_model() {
    local repo="$1"
    local filename="$2"
    
    echo "Downloading $filename from $repo..."
    
    if [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
        wget --header="Authorization: Bearer $HUGGING_FACE_HUB_TOKEN" \
             "https://huggingface.co/$repo/resolve/main/$filename" \
             -O "$MODEL_DIR/$filename"
    else
        wget "https://huggingface.co/$repo/resolve/main/$filename" \
             -O "$MODEL_DIR/$filename"
    fi
    
    echo "Downloaded $filename successfully!"
}

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Default models to download
case "${1:-default}" in
    "llama2-7b")
        download_model "TheBloke/Llama-2-7B-Chat-GGUF" "llama-2-7b-chat.Q4_K_M.gguf"
        ln -sf "$MODEL_DIR/llama-2-7b-chat.Q4_K_M.gguf" "$MODEL_DIR/current-model.gguf"
        ;;
    "llama2-13b")
        download_model "TheBloke/Llama-2-13B-Chat-GGUF" "llama-2-13b-chat.Q4_K_M.gguf"
        ln -sf "$MODEL_DIR/llama-2-13b-chat.Q4_K_M.gguf" "$MODEL_DIR/current-model.gguf"
        ;;
    "codellama")
        download_model "TheBloke/CodeLlama-7B-Instruct-GGUF" "codellama-7b-instruct.Q4_K_M.gguf"
        ln -sf "$MODEL_DIR/codellama-7b-instruct.Q4_K_M.gguf" "$MODEL_DIR/current-model.gguf"
        ;;
    "mistral")
        download_model "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        ln -sf "$MODEL_DIR/mistral-7b-instruct-v0.1.Q4_K_M.gguf" "$MODEL_DIR/current-model.gguf"
        ;;
    "default")
        echo "Downloading default model (Llama 2 7B Chat)..."
        download_model "TheBloke/Llama-2-7B-Chat-GGUF" "llama-2-7b-chat.Q4_K_M.gguf"
        ln -sf "$MODEL_DIR/llama-2-7b-chat.Q4_K_M.gguf" "$MODEL_DIR/current-model.gguf"
        ;;
    *)
        echo "Usage: $0 [llama2-7b|llama2-13b|codellama|mistral|default]"
        exit 1
        ;;
esac

echo "Model setup complete! Current model: $(readlink $MODEL_DIR/current-model.gguf)"
EOF

chmod +x ~/ai-workspace/tools/download-model.sh

# Create llama.cpp wrapper script
cat > ~/ai-workspace/tools/llama-chat.sh << 'EOF'
#!/bin/bash
# Interactive chat with llama.cpp

MODEL_DIR="$HOME/ai-workspace/models/gguf"
CURRENT_MODEL="$MODEL_DIR/current-model.gguf"

if [[ ! -f "$CURRENT_MODEL" ]]; then
    echo "No model found. Please download a model first:"
    echo "  ~/ai-workspace/tools/download-model.sh"
    exit 1
fi

echo "Starting chat with $(basename $(readlink $CURRENT_MODEL))..."
echo "Type 'exit' to quit."

/usr/local/bin/llama-main \
    --model "$CURRENT_MODEL" \
    --interactive \
    --ctx-size 4096 \
    --threads 8 \
    --n-gpu-layers 35 \
    --color \
    --temp 0.7 \
    --repeat-penalty 1.1
EOF

chmod +x ~/ai-workspace/tools/llama-chat.sh

# Add llama.cpp aliases to bashrc
cat >> ~/.bashrc << 'EOF'

# llama.cpp aliases
alias llama-chat='~/ai-workspace/tools/llama-chat.sh'
alias llama-download='~/ai-workspace/tools/download-model.sh'
alias llama-start='sudo systemctl start llama-server'
alias llama-stop='sudo systemctl stop llama-server'
alias llama-status='sudo systemctl status llama-server'
alias llama-logs='journalctl -u llama-server -f'
EOF

echo "llama.cpp installation complete!"
echo "Next steps:"
echo "1. Download a model: ~/ai-workspace/tools/download-model.sh"
echo "2. Start interactive chat: llama-chat"
echo "3. Start server: llama-start"