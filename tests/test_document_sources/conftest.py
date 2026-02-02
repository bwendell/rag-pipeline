"""Shared fixtures for document source tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_directory(tmp_path: Path) -> Path:
    """Create a sample directory structure for testing.

    Structure:
        sample/
        ├── src/
        │   ├── main.py
        │   ├── utils.py
        │   └── __pycache__/
        │       └── main.cpython-311.pyc
        ├── docs/
        │   ├── README.md
        │   └── guide.md
        ├── tests/
        │   └── test_main.py
        ├── .git/
        │   └── config
        ├── .env
        └── requirements.txt
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__pycache__").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".git").mkdir()

    (tmp_path / "src/main.py").write_text('def main():\n    print("Hello, World!")\n')
    (tmp_path / "src/utils.py").write_text("def helper():\n    return 42\n")
    (tmp_path / "src/__pycache__/main.cpython-311.pyc").write_bytes(b"\x00" * 100)
    (tmp_path / "tests/test_main.py").write_text("def test_main():\n    assert True\n")
    (tmp_path / "docs/README.md").write_text("# Project\n\nDescription here.\n")
    (tmp_path / "docs/guide.md").write_text("# Guide\n\n## Section 1\n\nContent.\n")
    (tmp_path / ".git/config").write_text("[core]\n  autocrlf = true\n")
    (tmp_path / ".env").write_text("SECRET_KEY=abc123\n")
    (tmp_path / "requirements.txt").write_text("fastapi>=0.100\nchromadb>=0.4\n")

    return tmp_path


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a single Python file for testing."""
    file_path = tmp_path / "sample.py"
    file_path.write_text('''"""Sample module."""

import os
from typing import List

class Calculator:
    """A simple calculator."""
    
    def add(self, a: int, b: int) -> int:
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        return a * b

def main() -> None:
    calc = Calculator()
    print(calc.add(2, 3))
''')
    return file_path
