"""
Entry point used when WDoc is imported or called by 'python -m WDoc'.
Does the same as __init__.py
"""


if __name__ == "__main__":
    from . import cli_launcher
    cli_launcher()
