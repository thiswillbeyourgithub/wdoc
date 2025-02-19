"""
easy access for all other files wether we are in verbose mode or not etc
"""

import os
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

is_silent = check_kwargs("silent", "s")

md_printing_disabled = check_kwargs("disable_md_printing")


class PrivateSanityChecker(int):
    "simple class that ALWAYS checks that WDOC_PRIVATE_MODE is appropriate when is_private is compared to anything"

    def __new__(cls, value=False):  # needed to subclass bool
        return super().__new__(cls, bool(value))

    def __init__(self, value):
        assert isinstance(value, bool)
        self.value = value

    def __sanity_check__(self):
        if self.value:
            assert str(os.environ["WDOC_PRIVATE_MODE"]).lower() == "true"
        else:
            assert str(os.environ["WDOC_PRIVATE_MODE"]).lower() == "false"

    def __eq__(self, other):
        self.__sanity_check__()
        return self.value == other


is_private = PrivateSanityChecker(check_kwargs("private"))

# useful to know if we should use tqdm or not (it can cause broken pipe errors
# otherwise) and modify the formatting output
is_piped = sys.stdout.isatty()
