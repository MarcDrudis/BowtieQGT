# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Bowtie Quantum Geometric Tensor"
copyright = "2025, IBM"
author = "Marc Drudis"
from bowtie_qgt import __version__ as release  # noqa: E402, F401  (read by Sphinx)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/Numpy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",  # Markdown, used to include CHANGELOG.md
]

# Autodoc settings. Members are listed explicitly in docs/source/api/*.rst, so
# automodule only renders each module's docstring and nothing is documented twice.
autodoc_default_options = {
    "member-order": "bysource",
}

# Autosummary settings
autosummary_generate = True

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping for cross-referencing external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer/", None),
}


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_nefertiti"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "project_short": "BowtieQGT",  # shown instead of the full name on narrow screens
    # The theme puts the logo in the header, just left of the project name.
    "logo": "bowtie_logo.svg",
    "logo_alt": "BowtieQGT logo",
    "logo_width": 56,  # keeps the logo's 200:124 aspect ratio
    "logo_height": 35,
}
