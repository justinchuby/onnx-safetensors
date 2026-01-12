# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Configuration file for the Sphinx documentation builder.

To run the documentation: python -m sphinx docs dist/html
"""

import sys

import onnx_safetensors

# -- Project information -----------------------------------------------------

project = "onnx-safetensors"
copyright = "Justin Chu. All rights reserved."
author = "ONNX Contributors"
version = onnx_safetensors.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "default"
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_css_files = []


# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}
