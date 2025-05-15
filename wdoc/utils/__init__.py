# Using a __init__.py to force the order of initialization:
# 1. load the env variables
# 2. Enable (or not) import tricks for faster startup time
from wdoc.utils import (
    batch_file_loader,
    customs,
    env,
    import_tricks,
    loaders,
    misc,
    prompts,
    tasks,
)

__all__ = [
    "env",
    "batch_file_loader",
    "loaders",
    "misc",
    "prompts",
    "tasks",
    "customs",
    "import_tricks",
]
