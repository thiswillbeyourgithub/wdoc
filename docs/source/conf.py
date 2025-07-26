# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# don't include docstrings of objects that are not part of wdoc
seen = []


def skip_imported(app, what, name, obj, skip, options):
    if hasattr(obj, "__module__"):
        # also make sure to only include each object once
        objid = id(obj)
        if objid in seen:
            return True
        else:
            seen.append(objid)
        if not obj.__module__.startswith("wdoc"):
            return True
    return skip


# Add the project root and extension directories to the Python path
sys.path.insert(0, os.path.abspath("./../.."))


def setup(app):
    # don't include docstrings of objects that are not part of wdoc
    app.connect("autodoc-skip-member", skip_imported)


project = "wdoc"
copyright = "2025, thiswillbeyourgithub"
author = "thiswillbeyourgithub"
release = "3.3.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",  # for markdown support
    "sphinx.ext.autosectionlabel",
]

# Autodoc settings
autodoc_default_options = {}

autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
# autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
# autodoc_docstring_signature = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True

exclude_patterns = []
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "language_selector": False,
    "style_external_links": True,
    "prev_next_buttons_location": "both",
    "analytics_anonymize_ip": True,
    #
    # 'navigation_depth': -1,
    # 'sidebar_hide_name': True,  # Less aggressive, just hides the project name
    #
    # # specific to pydata theme
    # # source: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/navigation.html
    "show_nav_level": 6,
    "collapse_navigation": True,
}
