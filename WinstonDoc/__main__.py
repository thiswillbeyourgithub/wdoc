"""
Entry point used when WinstonDoc is imported or called by 'python -m WinstonDoc'.
Does the same as __init__.py
"""

from . import cli_launcher

if __name__ == "__main__":
    cli_launcher()
