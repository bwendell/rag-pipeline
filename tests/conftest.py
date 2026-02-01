"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides fixtures
available to all tests in the test suite.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Will import types when implemented: from rag_pipeline.core.types import Document, Chunk


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir(project_root: Path) -> Path:
    """Return the test fixtures directory."""
    fixtures = project_root / "tests" / "fixtures"
    fixtures.mkdir(exist_ok=True)
    return fixtures


# =============================================================================
# Sample Content Fixtures
# =============================================================================

@pytest.fixture
def sample_python_code() -> str:
    """Return sample Python code for testing code chunking."""
    return '''"""
Sample module for testing.
"""
from typing import Optional


class Calculator:
    """A simple calculator class."""
    
    def __init__(self, precision: int = 2):
        """Initialize the calculator.
        
        Args:
            precision: Number of decimal places for results.
        """
        self.precision = precision
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: First number.
            b: Second number.
            
        Returns:
            The sum of a and b.
        """
        return round(a + b, self.precision)
    
    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide two numbers.
        
        Args:
            a: Dividend.
            b: Divisor.
            
        Returns:
            The quotient, or None if b is zero.
        """
        if b == 0:
            return None
        return round(a / b, self.precision)


def greet(name: str) -> str:
    """Return a greeting message.
    
    Args:
        name: The name to greet.
        
    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
'''


@pytest.fixture
def sample_markdown() -> str:
    """Return sample Markdown content for testing markdown chunking."""
    return '''# Project Documentation

This is the main documentation for the project.

## Installation

Follow these steps to install the project.

### Prerequisites

- Python 3.11+
- pip

### Steps

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

```bash
pip install -e ".[dev]"
```

## Configuration

The project uses environment variables for configuration.

### Required Variables

| Variable | Description |
|----------|-------------|
| `API_KEY` | Your API key |
| `DEBUG` | Enable debug mode |

## API Reference

### Endpoints

#### GET /health

Returns the health status of the service.

#### POST /query

Query the RAG pipeline.

**Request Body:**

```json
{
    "question": "What is this about?"
}
```

**Response:**

```json
{
    "answer": "This is about...",
    "sources": []
}
```
'''


@pytest.fixture
def sample_runbook() -> str:
    """Return sample runbook content for testing."""
    return '''# Incident Response: High CPU Usage

## Overview

This runbook describes how to respond to high CPU usage alerts.

## Severity

- **Critical**: CPU > 95% for 5 minutes
- **Warning**: CPU > 80% for 10 minutes

## Steps

### 1. Identify the Process

```bash
top -o %CPU
```

Look for processes using excessive CPU.

### 2. Check for Known Issues

- Check recent deployments
- Check for scheduled jobs
- Review application logs

### 3. Mitigation

#### Option A: Restart Service

```bash
systemctl restart my-service
```

#### Option B: Scale Out

```bash
kubectl scale deployment my-app --replicas=5
```

### 4. Escalation

If the issue persists after 15 minutes:
1. Page on-call engineer
2. Open incident ticket
3. Update status page

## Post-Incident

- Document root cause
- Create follow-up tickets
- Schedule retrospective
'''


# =============================================================================
# Mock Component Fixtures (To be expanded in later phases)
# =============================================================================

@pytest.fixture
def mock_embedding() -> list[float]:
    """Return a mock embedding vector (384 dimensions)."""
    return [0.1] * 384


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config() -> dict:
    """Return test configuration values."""
    return {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "mistral",
        "top_k": 5,
    }
