# AI Coding Guidelines for PV-Batch-Comparator

These guidelines must be strictly followed for all code contributions to ensure consistency, readability, and maintainability.

## 1. Code Style & Formatting

- **PEP 8**: All code must adhere to [PEP 8](https://peps.python.org/pep-0008/) standards.
- **Formatter**: Code must be compatible with **Black**.
    - **Line Length**: Limit lines to **88 characters**.
- **Imports**:
    - Use **absolute imports** (e.g., `from src.config import ...`) instead of relative imports.
    - Sort imports using **isort**.
    - Grouping order: Standard library -> Third-party -> Local application.

## 2. Type Hinting (CRITICAL)

- **Mandatory Hints**: All functions, methods, and class attributes must have Python 3 type hints.
- **Typing Module**: Use `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union`, etc., for complex types.
- **Return Types**: Always specify return types, even if `None`.

### Example
```python
from typing import List, Dict, Optional

def calculate_efficiency(voc: float, jsc: float, ff: float) -> float:
    """
    Calculate solar cell efficiency.
    
    Args:
        voc: Open circuit voltage (V)
        jsc: Short circuit current density (mA/cm²)
        ff: Fill factor (%)
        
    Returns:
        Efficiency (%)
    """
    return (voc * jsc * ff) / 100.0
```

## 3. Documentation

- **Docstrings**: All modules, classes, and functions must have docstrings.
- **Style**: Use **Google Style** docstrings (as shown in the example above).
- **Comments**: Use comments to explain *why* something is done, not *what* is done (unless complex).

## 4. Tooling

The project is configured to use the following tools to enforce these guidelines:

- **Black**: For code formatting.
- **isort**: For import sorting.
- **MyPy**: For static type checking.

### Running Checks
```bash
# Format code
black .
isort .

# Check types
mypy .
```
