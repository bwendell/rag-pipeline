#!/bin/bash
# =============================================================================
# RAG Pipeline - Development Setup Script
# =============================================================================
# This script sets up the development environment for the RAG Pipeline project.
# Usage: ./scripts/setup.sh
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  RAG Pipeline - Development Setup"
echo "=============================================="
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d ' ' -f 2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d '.' -f 1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d '.' -f 2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}Error: Python 3.11+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install package with dev dependencies
echo -e "${YELLOW}Installing package with dev dependencies...${NC}"
pip install -e ".[dev]" > /dev/null
echo -e "${GREEN}✓ Package installed${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/chroma
mkdir -p logs
echo -e "${GREEN}✓ Data directories created${NC}"

# Check for Ollama (optional)
echo ""
echo -e "${YELLOW}Checking for Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is running${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama is not running. Start with: ollama serve${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Ollama not found. Install from: https://ollama.ai/${NC}"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Start Ollama (if using local LLM):"
echo "     ollama serve"
echo "     ollama pull mistral"
echo ""
echo "  3. Run the development server:"
echo "     uvicorn rag_pipeline.main:app --reload"
echo ""
echo "  4. Run tests:"
echo "     pytest"
echo ""
