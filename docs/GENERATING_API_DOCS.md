# Generating API Documentation

This guide explains how to generate and maintain the API documentation for BowtieQGT.

## Prerequisites

Install the package with the `docs` extra (Sphinx, theme and extensions):

```bash
uv pip install -e ".[docs]"
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
├── examples.rst              # Usage examples (includes docs/examples/*.py)
├── changelog.rst             # Renders CHANGELOG.md with myst-parser
└── _static/                  # Logo, custom.css and the example plot
```

### The example plot

`docs/source/_static/mclachlan_ising.png` is committed, not generated during the build: the
script takes several minutes. After changing `docs/examples/mclachlan_ising.py`,
regenerate it and commit the new image:

```bash
uv pip install -e ".[examples]"
cd docs/source/_static && python ../../examples/mclachlan_ising.py
```

## How API Documentation Works

### Sphinx Autodoc

The API documentation is automatically generated from Python docstrings using Sphinx's `autodoc` extension. The configuration in `conf.py` includes:

- **autodoc**: Extracts documentation from Python docstrings
- **autosummary**: Generates summary tables of modules/classes/functions
- **napoleon**: Parses Google and NumPy style docstrings
- **viewcode**: Adds links to highlighted source code
- **intersphinx**: Creates links to external documentation (Qiskit, Qiskit Aer, NumPy, Python)
- **myst_parser**: Parses Markdown, used to include `CHANGELOG.md`

### Adding New Modules

To document a new module:

1. Create a new `.rst` file in `docs/source/api/`. `automodule` renders only the module
   docstring; list each public object once with its own directive, so nothing is
   documented twice:

```rst
module_name
===========

.. automodule:: bowtie_qgt.module_name

.. autofunction:: bowtie_qgt.module_name.some_function
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

An object is documented twice, usually because it is listed both through an `automodule`
with `:members:` and through its own `autofunction`/`autoclass`. Keep only one of the two.

## Continuous Integration

`.github/workflows/docs.yml` builds the docs on every pull request and deploys them to
GitHub Pages on pushes to `main`. To reproduce its build locally:

```bash
uv pip install -e ".[docs]"
cd docs
make clean html
```

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Sphinx Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
