import fire
import platform

# parse args again to know globally if we're in verbose mode
kwargs = fire.Fire(lambda *args, **kwargs: kwargs)
if "debug" in kwargs and kwargs["debug"]:
    is_verbose = True
    is_linux = platform.system() == "Linux"
else:
    is_verbose = False
    is_linux = False
