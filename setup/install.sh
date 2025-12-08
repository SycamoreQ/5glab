#!/bin/bash
# Install dependencies on all nodes

echo "Installing Ray Distributed System Dependencies..."

# Update system
sudo apt-get update

# Python dependencies
pip install --upgrade pip
pip install ray[default]==2.52.0
pip install gradio==6.0.2
pip install ollama
pip install neo4j==6.0.3
pip install sentence-transformers==5.1.2
pip install torch torchvision torchaudio
pip install numpy pandas pyyaml
pip install psutil

# Install Ollama (for LLM frontend)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Ollama model
ollama pull llama3.2

echo "Installation complete!"
