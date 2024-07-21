"""
easy access for all other files wether we are in verbose mode or not
"""

import fire
import platform

# parse args again to know globally if we're in verbose mode
kwargs = fire.Fire(lambda *args, **kwargs: kwargs)
is_linux = platform.system() == "Linux"

def check_kwargs(arg):
    if arg in kwargs and kwargs[arg]:
        return True
    return False

is_debug = check_kwargs("debug")
is_verbose = is_debug or check_kwargs("verbose")

is_silent = check_kwargs("silent")

md_printing_disabled = check_kwargs("disable_md_printing")
