# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(2, os.path.abspath('../../wdoc'))

project = 'wdoc'
copyright = '2024, thiswillbeyourgithub'
author = 'thiswillbeyourgithub'
release = '2.4.13'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser'  # for markdown support
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.css': 'css',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'flyout_display': 'attached',
    'language_selector': False,
    'style_external_links': True,
    'prev_next_buttons_location': 'both',
    'analytics_anonymize_ip': True,

    'navigation_depth': -1,
    'sidebar_hide_name': True,  # Less aggressive, just hides the project name

    # specific to pydata theme
    # source: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/navigation.html
    "show_nav_level": 6,
    "collapse_navigation": True

}
html_static_path = ['_static']
html_css_files = ["custom.css"]
