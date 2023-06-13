# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "discotime"
copyright = "2023, Peter Christoffer Holm"
author = "Peter Christoffer Holm"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "lightning": ("https://lightning.ai/docs/stable/", None),
    "lightning.pytorch": ("https://lightning.ai/docs/pytorch/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "exclude-members": "__weakref__",
}
nitpick_ignore = [
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "lightning.pytorch.core.datamodule.LightningDataModule"),
    ("py:class", "lightning.fabric.loggers.logger.Logger"),
    ("py:class", "lightning.pytorch.core.module.LightningModule"),
    ("py:class", "L.Fabric"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "discotime.datasets.utils.T_co"),
    ("py:class", "discotime.utils.misc.T"),
    ("py:class", "T"),
    ("py:class", "numpy.ndarray[typing.Any, numpy.dtype[+T_co]]"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy._typing._array_like._SupportsArray"),
    ("py:class", "numpy._typing._nested_sequence._NestedSequence"),
    ("py:class", "npt.ArrayLike"),
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
