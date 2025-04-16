"""
easy access for all other files wether we are in verbose mode or not etc
"""

import platform
import sys

is_linux = platform.system() == "Linux"

cmdline = " ".join(sys.argv[1:])


def check_kwargs(arg: str, abbrv: str = None) -> bool:
    if f" {arg}" in cmdline or f" --{arg}" in cmdline:
        return True
    if abbrv and f" -{abbrv}" in cmdline:
        return True
    return False


is_debug = check_kwargs("debug", "d")

is_verbose = is_debug or check_kwargs("verbose", "v")

md_printing_disabled = check_kwargs("disable_md_printing")

is_private = check_kwargs("private")

# useful to know if we should use tqdm or not (it can cause broken pipe errors
# otherwise) and modify the formatting output
is_piped = sys.stdout.isatty()
