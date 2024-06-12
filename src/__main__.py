"""
Entry point used when DocToolsLLM is imported or called by 'python -m DocToolsLLM'.
Does the same as __init__.py
"""

from . import cli_launcher

if __name__ == "__main__":
    cli_launcher()
