# Generating API Documentation

This guide explains how to generate and maintain the API documentation for BowtieQGT.

## Prerequisites

Ensure you have the documentation dependencies installed:

```bash
uv sync --all-extras
```

Or install the dev dependency group:

```bash
uv pip install -e ".[dev]"
```

## Building the Documentation

### Full Build

To build the complete HTML documentation:

```bash
cd docs
uv run sphinx-build -b html source build/html
```

The generated documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view it.

### Clean Build

To perform a clean build (removes all cached files):

```bash
cd docs
rm -rf build/
uv run sphinx-build -b html source build/html
```

### Using Make (Linux/macOS)

Alternatively, you can use the provided Makefile:

```bash
cd docs
make html
```

To clean and rebuild:

```bash
cd docs
make clean html
```

## Documentation Structure

The API documentation is organized as follows:

```
docs/source/
├── api.rst                    # Main API reference page
├── api/
│   ├── bowtieqgt.rst         # BowtieQGT class documentation
│   └── bowtie_circuits.rst   # Circuit utilities documentation
├── index.rst                  # Documentation home page
├── installation.rst           # Installation guide
├── quickstart.rst            # Quick start guide
└── examples.rst              # Usage examples
```

## How API Documentation Works

### Sphinx Autodoc

The API documentation is automatically generated from Python docstrings using Sphinx's `autodoc` extension. The configuration in `conf.py` includes:

- **autodoc**: Extracts documentation from Python docstrings
- **autosummary**: Generates summary tables of modules/classes/functions
- **napoleon**: Parses Google and NumPy style docstrings
- **viewcode**: Adds links to highlighted source code
- **intersphinx**: Creates links to external documentation (Qiskit, NumPy, Python)

### Adding New Modules

To document a new module:

1. Create a new `.rst` file in `docs/source/api/`:

```rst
module_name
===========

.. automodule:: bowtie_qgt.module_name
   :members:
   :undoc-members:
   :show-inheritance:
```

2. Add it to the toctree in `docs/source/api.rst`:

```rst
.. toctree::
   :maxdepth: 2

   api/bowtieqgt
   api/bowtie_circuits
   api/module_name
```

### Docstring Style

The project uses Google-style docstrings. Example:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description providing more details about what the function does,
    its behavior, and any important notes.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    pass
```

## Viewing the Documentation Locally

After building, you can view the documentation by opening the HTML files:

```bash
# Linux/macOS
open docs/build/html/index.html

# Or use Python's built-in HTTP server
cd docs/build/html
python -m http.server 8000
# Then visit http://localhost:8000 in your browser
```

## Troubleshooting

### Import Errors

If Sphinx cannot import your modules, ensure:

1. The project root is in the Python path (configured in `conf.py`)
2. All dependencies are installed
3. The package is installed in development mode: `uv pip install -e .`

### Missing Documentation

If functions/classes don't appear in the docs:

1. Check that they have docstrings
2. Verify they're not excluded in `autodoc_default_options`
3. Ensure the module is properly imported in `__init__.py`

### Warnings About Duplicate Objects

These warnings occur when autosummary generates duplicate entries. They're generally harmless but can be suppressed by adjusting the RST files to avoid redundant autodoc directives.

## Continuous Integration

The documentation build can be integrated into CI/CD pipelines:

```bash
# In your CI script
uv sync --all-extras
cd docs
uv run sphinx-build -W -b html source build/html
```

The `-W` flag treats warnings as errors, ensuring documentation quality.

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Sphinx Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)