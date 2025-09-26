# 🚀 Omarchy AI Quick Start Guide

Get up and running with your AI development environment in 15 minutes!

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] Fresh Arch Linux installation (minimum)
- [ ] 16GB RAM (32GB recommended)
- [ ] 100GB free disk space (500GB recommended)
- [ ] Internet connection
- [ ] NVIDIA GPU (optional but recommended)

## 🎯 Installation (5 minutes)

### Step 1: One-Command Installation
```bash
curl -fsSL https://raw.githubusercontent.com/mitkox/omarchy-ai/main/boot.sh | bash
```

This single command will:
- Clone the Omarchy AI repository
- Validate your system requirements
- Install all necessary components
- Configure your development environment

### Step 2: Wait for Installation
The installation will take 10-30 minutes depending on your system. You'll see progress indicators like:
```
[1/12] (8%) Validating system requirements
[2/12] (16%) Installing base Omarchy system
[3/12] (25%) Setting up AI development tools
...
```

### Step 3: Reboot
After installation completes:
```bash
sudo reboot
```

## ✅ Verification (2 minutes)

### Check Installation Health
```bash
# Run system diagnostics
omarchy-ai-doctor

# Quick functionality test
omarchy-ai-test --quick
```

You should see:
```
✅ System appears to be healthy!
✅ Quick health check passed
```

## 🛠️ First Steps (8 minutes)

### 1. Activate AI Environment (30 seconds)
```bash
# Activate the AI development environment
ai-env
```

You should see:
```
✅ AI development environment activated!
🐍 Python: Python 3.11.x
📦 Conda env: ai-dev
```

### 2. Explore Your AI Workspace (1 minute)
```bash
# Navigate to AI workspace
ai-workspace

# See what's available
ls -la
```

Your workspace contains:
```
projects/      # Your AI projects
models/        # Downloaded AI models
datasets/      # Training and test data
experiments/   # ML experiment tracking
notebooks/     # Jupyter notebooks
logs/          # Application logs
```

### 3. Test Python Environment (1 minute)
```bash
# Test core libraries
python -c "
import torch
import transformers
import numpy as np
print('✅ Core libraries working!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 4. Start Jupyter Lab (30 seconds)
```bash
# Start Jupyter Lab in AI workspace
jupyter-ai
```

Your browser will open to Jupyter Lab interface.

### 5. Download Your First Model (3 minutes)
```bash
# Download a small conversational model
model-download microsoft/DialoGPT-small

# List downloaded models
model-list
```

### 6. Create Your First AI Notebook (2 minutes)

In Jupyter Lab, create a new notebook and try:

```python
# Cell 1: Test basic functionality
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

print("🤖 Omarchy AI is ready!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Cell 2: Load a model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print("✅ Model loaded successfully!")

# Cell 3: Generate text
def chat_with_ai(message):
    inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(message)[1] if message in response else response

# Test the model
response = chat_with_ai("Hello, how are you?")
print(f"AI: {response}")
```

### 7. Monitor System Resources (30 seconds)
```bash
# In a new terminal, monitor GPU usage (if available)
gpu-monitor

# Or check basic system stats
htop
```

## 🎯 Next Steps - Choose Your Path

### For ML Engineers
```bash
# Start MLflow for experiment tracking
mlflow-ui

# Create a new project
mkdir ~/ai-workspace/projects/my-first-project
cd ~/ai-workspace/projects/my-first-project

# Initialize git repository
git init
```

### For Data Scientists
```bash
# Download a dataset
dataset-download squad

# Start TensorBoard
tensorboard-ai

# Create analysis notebook
jupyter-ai
```

### For AI Researchers
```bash
# Download larger models
model-download microsoft/DialoGPT-medium
model-download microsoft/DialoGPT-large

# Set up distributed training
# (Advanced - see documentation)
```

## 🔧 Essential Commands Reference

### Environment Management
```bash
ai-env                    # Activate AI environment
ai-workspace             # Go to AI workspace
conda deactivate         # Exit environment
```

### Model Management
```bash
model-download <model>   # Download from Hugging Face
model-list              # List downloaded models  
model-info <model>      # Show model details
model-serve             # Start model API server
model-cleanup           # Clean up cache
```

### Development Tools
```bash
jupyter-ai              # Start Jupyter Lab
mlflow-ui               # Start MLflow UI
tensorboard-ai          # Start TensorBoard
docs-serve              # Start documentation server
```

### System Tools
```bash
omarchy-ai-doctor       # Run diagnostics
omarchy-ai-repair       # Fix common issues
omarchy-ai-test         # Run test suite
gpu-monitor             # Monitor GPU usage
```

## 🎮 Try These Examples

### Example 1: Text Generation
```python
from transformers import pipeline

# Create a text generation pipeline
generator = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Generate text
result = generator("The future of AI is", max_length=50)
print(result[0]['generated_text'])
```

### Example 2: Sentiment Analysis
```python
from transformers import pipeline

# Create sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Analyze sentiment
result = classifier("I love using Omarchy AI!")
print(result)
```

### Example 3: Question Answering
```python
from transformers import pipeline

# Create QA pipeline
qa = pipeline("question-answering")

context = "Omarchy AI is a development environment for AI engineers."
question = "What is Omarchy AI?"

result = qa(question=question, context=context)
print(f"Answer: {result['answer']}")
```

## 🎯 15-Minute Challenge Projects

### Project 1: Personal Chatbot
1. Download a conversational model
2. Create a simple chat interface
3. Add memory for conversation history
4. Deploy as a local web service

### Project 2: Document Q&A System
1. Load a document dataset
2. Create embeddings for documents
3. Build a question-answering system
4. Create a Jupyter notebook demo

### Project 3: Image Classifier
1. Download an image classification model
2. Create a simple image upload interface
3. Classify uploaded images
4. Display results with confidence scores

## 🆘 Quick Troubleshooting

### Issue: "ai-env command not found"
```bash
# Reload shell configuration
source ~/.bashrc

# Or manually activate
conda activate ai-dev
```

### Issue: "Model download failed"
```bash
# Check network connection
curl -I https://huggingface.co

# Try with proxy or mirror if needed
export HF_ENDPOINT=https://hf-mirror.com
```

### Issue: "GPU not detected"
```bash
# Check GPU
nvidia-smi

# Reinstall drivers if needed
sudo pacman -S nvidia nvidia-utils
```

### Issue: "Jupyter won't start"
```bash
# Reinstall Jupyter
conda activate ai-dev
pip install --force-reinstall jupyter jupyterlab
```

## 🚀 Advanced Quick Starts

### GPU Acceleration Setup
```bash
# Test GPU functionality
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### Container Development
```bash
# Start development containers
containers start ai-jupyter

# Access containerized environment
docker exec -it ai-jupyter bash
```

### Distributed Training
```bash
# Initialize distributed training
distributed-train my_training_script.py --nodes 1 --gpus 1
```

## 📚 Learning Resources

### Documentation
- `~/ai-workspace/docs/` - Offline documentation
- `docs-serve` - Start documentation server
- README.md - Project overview
- TROUBLESHOOTING.md - Problem solutions

### Example Notebooks
```bash
# Copy example notebooks
cp -r ~/.local/share/omarchy-ai/examples ~/ai-workspace/notebooks/

# Open examples in Jupyter
jupyter-ai
```

### Community
- GitHub Issues: Report bugs or ask questions
- Discussions: Share projects and get help
- Wiki: Community-contributed guides

## 🎉 Congratulations!

You now have a fully functional AI development environment! Here's what you've accomplished:

✅ Installed a complete AI development stack  
✅ Activated your first AI environment  
✅ Downloaded and tested an AI model  
✅ Created your first AI notebook  
✅ Learned essential commands  

### What's Next?
1. **Explore**: Try the example projects above
2. **Learn**: Work through the included tutorials
3. **Build**: Start your own AI project
4. **Share**: Contribute back to the community

### Keep Learning
- Experiment with different models
- Try various AI tasks (NLP, computer vision, etc.)
- Set up experiment tracking with MLflow
- Deploy your models locally
- Contribute improvements to Omarchy AI

### Need Help?
- Run `omarchy-ai-doctor` for health checks
- Check `TROUBLESHOOTING.md` for solutions
- Ask questions in GitHub issues
- Join the community discussions

Happy AI developing! 🤖✨