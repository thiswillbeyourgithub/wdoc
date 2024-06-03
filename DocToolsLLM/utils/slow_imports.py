"""
litellm is very slow to import and is always nedded. So here,
we do a lazy loading and start importing it in a separate thread.

Unstructured can be very slow too but is not always needed
"""
import sys
import lazy_import

# from multiprocessing import Process as Parallel
from threading import Thread as Parallel


IMPORT_DEBUG = 1
if IMPORT_DEBUG:
    def p(message): print(message)
else:
    def p(message): pass

to_imports = [
    "litellm",
    # "unstructured",
]

def importer(name) -> None:
    p(f"Importing {name}")
    exec(f"import {name}")

    p(f"Triggering non lazy loading of {name}")
    eval(f"dir({name})")

    p(f"Sending {name}")
    imported[to_imports.index(name)] = locals()[name]
    return

class parallel_importer:
    def __init__(self, name):
        name = name
        process = Parallel(
            target=importer,
            args=(name,),
            daemon=False,
        )
        process.start()
        process.join()
        setattr_queue = {}
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "setattr_queue", setattr_queue)

    def __getattr__(self, name):
        instance_name = object.__getattribute__(self, "name")
        process = object.__getattribute__(self, "process")
        setattr_queue = object.__getattribute__(self, "setattr_queue")

        if not isinstance(imported[to_imports.index(instance_name)], parallel_importer):
            p(f"{instance_name}: get {name}")
            new = imported[to_imports.index(instance_name)]
        elif process.is_alive():
            p(f"{instance_name}: joining")
            process.join()
            assert isinstance(imported[to_imports.index(instance_name)], parallel_importer)
        else:
            new = imported[to_imports.index(instance_name)]
        for qn, qv in object.__getattribute__(self, "setattr_queue").items():
            setattr(new, qn, qv)
        setattr_queue.clear()

        sys.modules[instance_name] = new
        return getattr(new, name)

    def __setattr__(self, name, value):
        instance_name = object.__getattribute__(self, "name")
        setattr_queue = object.__getattribute__(self, "setattr_queue")
        p(f"{instance_name.upper()}:SETATTR: {name}")

        if isinstance(imported[to_imports.index(instance_name)], parallel_importer):
            setattr_queue[name] = value
        else:
            new = imported[to_imports.index(instance_name)]
            for qn, qv in setattr_queue.items():
                setattr(new, qn, qv)
            setattr_queue.clear()


imported = [None for i in to_imports]

for it, to_do in enumerate(to_imports):
    imported[it] = parallel_importer(name=to_do)
