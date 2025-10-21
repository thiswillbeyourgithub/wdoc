from typing import List
from dataclasses import dataclass, field

__valid_tasks__: List[str] = [
    "query",
    "summarize",
    "parse",
    "search",
    "summarize_then_query",
]


@dataclass
class wdocTask:
    original: str
    query: bool = field(init=False)
    summarize: bool = field(init=False)
    parse: bool = field(init=False)
    search: bool = field(init=False)

    def __post_init__(self):
        # default values
        self.query = False
        self.summarize = False
        self.parse = False
        self.search = False

        # checks
        if isinstance(self.original, wdocTask):
            self.original = self.original.original
        self.original = self.original.replace("summary", "summarize")
        assert (
            self.original in __valid_tasks__
        ), f"Received task '{self.original}' is not part of expected tasks: '{__valid_tasks__}'"

        # set the actual properties
        if "query" in self.original:
            self.query = True
        if "summarize" in self.original:
            self.summarize = True
        if "search" in self.original:
            self.search = True
        if "parse" in self.original:
            self.parse = True

    def __hash__(self):
        "necessary for memoizing"
        return self.original.__hash__()

    def __str__(self) -> str:
        return self.original
