# Offline Documentation Setup for Omarchy AI
# Downloads and sets up offline documentation for AI development

# Create documentation directories
mkdir -p ~/ai-workspace/docs/{python,pytorch,tensorflow,huggingface,llama-cpp,general}
mkdir -p ~/ai-workspace/docs/cheatsheets
mkdir -p ~/ai-workspace/docs/references

# Install documentation tools
pip install \
  pdoc \
  sphinx \
  mkdocs \
  mkdocs-material \
  pydoc-markdown \
  dash \
  dash-bootstrap-components

# Download Python documentation
echo "Downloading Python documentation..."
cd ~/ai-workspace/docs/python
wget -r --no-parent --no-host-directories --cut-dirs=2 \
  https://docs.python.org/3/download.html \
  -A "*.zip" -P downloads/
if [ -f downloads/python-*.zip ]; then
  unzip downloads/python-*.zip
  rm downloads/python-*.zip
fi

# Download PyTorch documentation
echo "Downloading PyTorch documentation..."
cd ~/ai-workspace/docs/pytorch
git clone https://github.com/pytorch/pytorch.git --depth 1
cd pytorch/docs
pip install -r requirements.txt
make html
mv build/html/* ../
cd ..
rm -rf pytorch

# Download TensorFlow documentation
echo "Downloading TensorFlow documentation..."
cd ~/ai-workspace/docs/tensorflow
git clone https://github.com/tensorflow/docs.git --depth 1
cd docs
pip install -r tools/requirements.txt
cd ..
rm -rf docs/.git

# Download Hugging Face documentation
echo "Downloading Hugging Face documentation..."
cd ~/ai-workspace/docs/huggingface
git clone https://github.com/huggingface/transformers.git --depth 1
cd transformers/docs
pip install -r requirements.txt
make html
mv build/html/* ../
cd ..
rm -rf transformers

# Download llama.cpp documentation
echo "Downloading llama.cpp documentation..."
cd ~/ai-workspace/docs/llama-cpp
git clone https://github.com/ggerganov/llama.cpp.git --depth 1
cp -r llama.cpp/docs/* .
rm -rf llama.cpp

# Create cheatsheets
echo "Creating AI/ML cheatsheets..."

# Python cheatsheet
cat > ~/ai-workspace/docs/cheatsheets/python-ml.md << 'EOF'
# Python Machine Learning Cheatsheet

## NumPy
```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
eye = np.eye(3)

# Array operations
arr.shape, arr.dtype, arr.size
arr.reshape(2, 2)
arr.sum(), arr.mean(), arr.std()
```

## Pandas
```python
import pandas as pd

# DataFrame creation
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df = pd.read_csv('file.csv')

# Data manipulation
df.head(), df.tail(), df.info(), df.describe()
df.groupby('column').sum()
df.merge(other_df, on='key')
```

## Scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Tensor operations
x = torch.tensor([1, 2, 3, 4])
x = torch.randn(3, 4)
x.shape, x.dtype, x.device

# Neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
EOF

# TensorFlow cheatsheet
cat > ~/ai-workspace/docs/cheatsheets/tensorflow.md << 'EOF'
# TensorFlow Cheatsheet

## Basic Operations
```python
import tensorflow as tf

# Tensors
x = tf.constant([1, 2, 3, 4])
x = tf.random.normal([3, 4])
x = tf.zeros([2, 3])

# Operations
y = tf.matmul(x, x)
y = tf.reduce_sum(x)
y = tf.reduce_mean(x)
```

## Keras
```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## Data Pipeline
```python
import tensorflow_datasets as tfds

# Load dataset
dataset = tfds.load('mnist', split='train')
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

dataset = dataset.map(augment)
```
EOF

# Hugging Face cheatsheet
cat > ~/ai-workspace/docs/cheatsheets/huggingface.md << 'EOF'
# Hugging Face Transformers Cheatsheet

## Basic Usage
```python
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Tokenize text
inputs = tokenizer("Hello world", return_tensors='pt')
outputs = model(**inputs)
```

## Text Generation
```python
from transformers import pipeline

# Create pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
outputs = generator("Hello, I'm a language model", 
                   max_length=50, 
                   num_return_sequences=1)
```

## Fine-tuning
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Common Tasks
```python
# Sentiment analysis
classifier = pipeline('sentiment-analysis')
result = classifier("I love this product!")

# Named entity recognition
ner = pipeline('ner', aggregation_strategy='simple')
entities = ner("My name is John and I live in New York")

# Question answering
qa = pipeline('question-answering')
answer = qa(question="What is my name?", context="My name is John")
```
EOF

# llama.cpp cheatsheet
cat > ~/ai-workspace/docs/cheatsheets/llama-cpp.md << 'EOF'
# llama.cpp Cheatsheet

## Basic Usage
```bash
# Interactive mode
./main -m model.gguf -p "Hello, my name is"

# Batch processing
./main -m model.gguf -f prompts.txt

# Server mode
./server -m model.gguf --host 0.0.0.0 --port 8080
```

## Common Parameters
```bash
# Model and context
-m model.gguf          # Model file
-c 4096               # Context size
-n 128                # Number of tokens to predict

# Sampling
-t 0.7                # Temperature
-p 0.9                # Top-p sampling
--top-k 40            # Top-k sampling
--repeat-penalty 1.1  # Repetition penalty

# Performance
--threads 8           # Number of threads
--n-gpu-layers 35     # GPU layers (CUDA)
--batch-size 512      # Batch size
```

## API Usage
```python
import requests

# Text generation
response = requests.post('http://localhost:8080/completion', 
                        json={
                            'prompt': 'Hello world',
                            'n_predict': 128,
                            'temperature': 0.7
                        })

# Chat completion
response = requests.post('http://localhost:8080/v1/chat/completions',
                        json={
                            'messages': [
                                {'role': 'user', 'content': 'Hello!'}
                            ],
                            'temperature': 0.7
                        })
```

## Model Quantization
```bash
# Convert to GGUF format
./quantize model.bin model.gguf q4_0

# Available quantization types
# q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32
```
EOF

# Create offline documentation server
cat > ~/ai-workspace/tools/docs-server.py << 'EOF'
#!/usr/bin/env python3
"""
Offline Documentation Server for Omarchy AI
Serves local documentation via web interface
"""

import os
import sys
from pathlib import Path
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import argparse
import threading
import time

class DocumentationServer:
    def __init__(self, docs_path: str = "~/ai-workspace/docs", port: int = 8080):
        self.docs_path = Path(docs_path).expanduser()
        self.port = port
        self.server = None
        
    def start_server(self):
        """Start the documentation server."""
        os.chdir(self.docs_path)
        
        handler = SimpleHTTPRequestHandler
        self.server = HTTPServer(("localhost", self.port), handler)
        
        print(f"Starting documentation server at http://localhost:{self.port}")
        print(f"Serving documentation from: {self.docs_path}")
        print("Press Ctrl+C to stop the server")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping documentation server...")
            self.server.shutdown()
    
    def open_browser(self):
        """Open browser to documentation."""
        time.sleep(1)  # Wait for server to start
        webbrowser.open(f"http://localhost:{self.port}")
    
    def run(self, open_browser: bool = True):
        """Run the documentation server."""
        if open_browser:
            browser_thread = threading.Thread(target=self.open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        self.start_server()

def main():
    parser = argparse.ArgumentParser(description="Offline Documentation Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--docs-path", default="~/ai-workspace/docs", help="Documentation path")
    
    args = parser.parse_args()
    
    server = DocumentationServer(args.docs_path, args.port)
    server.run(not args.no_browser)

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ai-workspace/tools/docs-server.py

# Create documentation index
cat > ~/ai-workspace/docs/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Omarchy AI Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            color: #333;
            margin-top: 0;
        }
        .card a {
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
        }
        .card a:hover {
            text-decoration: underline;
        }
        .card ul {
            list-style-type: none;
            padding-left: 0;
        }
        .card li {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .card li:last-child {
            border-bottom: none;
        }
        .status {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .online { background-color: #4CAF50; }
        .offline { background-color: #f44336; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Omarchy AI Documentation</h1>
        <p>Offline AI Development Resources</p>
        <span class="status offline">OFFLINE MODE</span>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🐍 Python Documentation</h3>
            <ul>
                <li><a href="python/">Python Official Docs</a></li>
                <li><a href="cheatsheets/python-ml.md">Python ML Cheatsheet</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🔥 PyTorch</h3>
            <ul>
                <li><a href="pytorch/">PyTorch Documentation</a></li>
                <li><a href="pytorch/tutorials/">Tutorials</a></li>
                <li><a href="pytorch/docs/">API Reference</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🧠 TensorFlow</h3>
            <ul>
                <li><a href="tensorflow/">TensorFlow Documentation</a></li>
                <li><a href="cheatsheets/tensorflow.md">TensorFlow Cheatsheet</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🤗 Hugging Face</h3>
            <ul>
                <li><a href="huggingface/">Transformers Documentation</a></li>
                <li><a href="cheatsheets/huggingface.md">Hugging Face Cheatsheet</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🦙 llama.cpp</h3>
            <ul>
                <li><a href="llama-cpp/">llama.cpp Documentation</a></li>
                <li><a href="cheatsheets/llama-cpp.md">llama.cpp Cheatsheet</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>📚 Cheatsheets</h3>
            <ul>
                <li><a href="cheatsheets/python-ml.md">Python ML</a></li>
                <li><a href="cheatsheets/tensorflow.md">TensorFlow</a></li>
                <li><a href="cheatsheets/huggingface.md">Hugging Face</a></li>
                <li><a href="cheatsheets/llama-cpp.md">llama.cpp</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🔧 Tools & References</h3>
            <ul>
                <li><a href="references/">API References</a></li>
                <li><a href="general/">General AI Resources</a></li>
            </ul>
        </div>

        <div class="card">
            <h3>🚀 Getting Started</h3>
            <ul>
                <li><a href="../PRD.md">Product Requirements</a></li>
                <li><a href="../README.md">Installation Guide</a></li>
            </ul>
        </div>
    </div>

    <script>
        // Add offline status indicator
        window.addEventListener('online', () => {
            document.querySelector('.status').textContent = 'ONLINE';
            document.querySelector('.status').className = 'status online';
        });

        window.addEventListener('offline', () => {
            document.querySelector('.status').textContent = 'OFFLINE MODE';
            document.querySelector('.status').className = 'status offline';
        });
    </script>
</body>
</html>
EOF

# Create documentation management script
cat > ~/ai-workspace/tools/docs-manager.sh << 'EOF'
#!/bin/bash
# Documentation Manager for Omarchy AI

DOCS_DIR="$HOME/ai-workspace/docs"
DOCS_SERVER="$HOME/ai-workspace/tools/docs-server.py"

case "$1" in
    "serve")
        echo "Starting documentation server..."
        python "$DOCS_SERVER" --port "${2:-8080}"
        ;;
    "update")
        echo "Updating documentation..."
        source ~/.local/share/omarchy-ai/install/offline-docs.sh
        ;;
    "search")
        if [ -z "$2" ]; then
            echo "Usage: $0 search <query>"
            exit 1
        fi
        echo "Searching documentation for: $2"
        find "$DOCS_DIR" -name "*.md" -o -name "*.html" -o -name "*.txt" | xargs grep -l "$2"
        ;;
    "open")
        if [ -z "$2" ]; then
            echo "Available documentation:"
            ls -la "$DOCS_DIR"
        else
            if [ -d "$DOCS_DIR/$2" ]; then
                xdg-open "$DOCS_DIR/$2"
            else
                echo "Documentation not found: $2"
            fi
        fi
        ;;
    *)
        echo "Documentation Manager for Omarchy AI"
        echo "Usage: $0 [serve|update|search|open]"
        echo ""
        echo "Commands:"
        echo "  serve [port]    - Start documentation server (default port: 8080)"
        echo "  update          - Update documentation"
        echo "  search <query>  - Search documentation"
        echo "  open [section]  - Open documentation section"
        ;;
esac
EOF

chmod +x ~/ai-workspace/tools/docs-manager.sh

# Add documentation aliases
cat >> ~/.bashrc << 'EOF'

# Documentation Aliases
alias docs='~/ai-workspace/tools/docs-manager.sh'
alias docs-serve='~/ai-workspace/tools/docs-manager.sh serve'
alias docs-update='~/ai-workspace/tools/docs-manager.sh update'
alias docs-search='~/ai-workspace/tools/docs-manager.sh search'
alias docs-open='~/ai-workspace/tools/docs-manager.sh open'
EOF

echo "Offline documentation setup complete!"
echo "Available commands:"
echo "  docs-serve          - Start documentation server"
echo "  docs-update         - Update documentation"
echo "  docs-search <query> - Search documentation"
echo "  docs-open [section] - Open documentation section"