"""
litellm is very slow to import and is always nedded. So here,
we do a lazy loading and start importing it in a separate thread.

Unstructured can be very slow too but is not always needed
"""
import sys
import lazy_import
from threading import Thread
from queue import Queue

IMPORT_DEBUG = False
if IMPORT_DEBUG:
    def p(message): print(message)
else:
    def p(message): pass

to_imports = [
    "litellm",
    # "unstructured",
]

def importer(q, name) -> None:
    p(f"Importing {name}")
    exec(f"import {name}")

    p(f"Triggering non lazy loading of {name}")
    # eval(f"dir({name})")

    p(f"Sending {name}")
    q.put(locals()[name])
    return


class threaded_importer:
    def __init__(self, name):
        self.name = name
        self.q = Queue()
        self.thread = Thread(
            target=importer,
            args=(self.q, self.name),
            daemon=False,
        )
        self.thread.start()
        self.setattr_queue = {}

    def __getattr__(self, name):
        if name == "thread":
            return super().__getattr__(name)

        assert self.thread.is_alive(), f"Failed: {self.name}"
        p(f"Requesting {self.name} that is still loading, stalling while I finish importing...")
        self.thread.join()

        new = self.q.get()
        for qn, qv in self.setattr_queue.items():
            setattr(new, qn, qv)

        sys.modules[self.name] = new
        globals()[self.name] = new
        p(f"Done threaded importing {self.name}")
        return getattr(new, name)

    def __setattr__(self, name, value):
        if name in ["q", "thread", "setattr_queue", "name"]:
            return super().__setattr__(name, value)
        else:
            assert self.thread.is_alive(), f"race condition: {self.name}"
            self.setattr_queue[name] = value


for to_do in to_imports:
    locals()[to_do] = lazy_import.lazy_module(to_do)

for to_do in to_imports:
    locals()[to_do] = threaded_importer(name=to_do)
