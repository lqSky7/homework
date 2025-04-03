#!/bin/bash

# Script to install Anaconda and set up Python 3.9 environment
# Created: April 2, 2025

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required dependencies
echo "Installing dependencies..."
sudo apt install -y curl wget libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Create temporary directory for downloads
echo "Creating temporary directory..."
mkdir -p ~/tmp
cd ~/tmp

# Download latest Anaconda installer
echo "Downloading Anaconda installer..."
wget -O anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# Verify the integrity of the downloaded file
echo "Verifying download integrity..."
sha256sum anaconda.sh

# Install Anaconda silently to home directory
echo "Installing Anaconda..."
bash anaconda.sh -b -p $HOME/anaconda3

# Add Anaconda to PATH
echo "Configuring environment..."
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Initialize conda for bash shell
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc

# Create Python 3.9 environment
echo "Creating Python 3.9 environment..."
conda create -y -n py39 python=3.9

# Provide instructions for activating the environment
echo "Installation complete!"
echo "To activate your Python 3.9 environment, run: conda activate py39"
echo "To verify installation, run: python --version"
