#!/bin/bash
# Install Tesseract OCR for typography evaluation pipeline

set -e

echo "Installing Tesseract OCR..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        sudo yum install -y epel-release
        sudo yum install -y tesseract tesseract-langpack-eng
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y tesseract tesseract-langpack-eng
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S --noconfirm tesseract tesseract-data-eng
    else
        echo "Unsupported Linux distribution. Please install Tesseract manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        brew install tesseract
    else
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh"
        exit 1
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash, Cygwin, etc.)
    echo "For Windows, please download Tesseract from:"
    echo "https://github.com/UB-Mannheim/tesseract/wiki"
    echo "Then add the installation directory to your PATH."
    exit 1
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Verify installation
if command -v tesseract &> /dev/null; then
    echo "Tesseract installed successfully!"
    tesseract --version
else
    echo "Tesseract installation may have failed. Please check manually."
    exit 1
fi
