"""
Entry point used when WDoc is imported or called by 'python -m WDoc'.
Does the same as __init__.py
"""

from . import cli_launcher

if __name__ == "__main__":
    cli_launcher()
