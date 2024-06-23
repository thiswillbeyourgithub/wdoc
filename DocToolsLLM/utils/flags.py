"""
easy access for all other files wether we are in verbose mode or not
"""

import fire
import platform

# parse args again to know globally if we're in verbose mode
kwargs = fire.Fire(lambda *args, **kwargs: kwargs)
is_linux = platform.system() == "Linux"
if "debug" in kwargs and kwargs["debug"]:
    is_debug = True
    is_verbose = True
else:
    is_debug = False
    is_verbose = False
