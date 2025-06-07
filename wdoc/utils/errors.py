"""
Exception classes
"""


class NoDocumentsRetrieved(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class NoDocumentsAfterLLMEvalFiltering(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ShouldIncreaseTopKAfterLLMEvalFiltering(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidDocEvaluationByLLMEval(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnexpectedDocDictArgument(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(
            f"You're trying to use an argument for a filetype that does not expect it: {message}"
        )


class TimeoutPdfLoaderError(Exception):
    def __init__(self) -> None:
        super().__init__()


class FrozenAttributeCantBeSet(AttributeError):
    """
    The attributes of EnvDataclass are frozen on purpose
    to avoid race condition as accessing an attribute loads
    the value dynamically from the environment.
    """

    def __init__(self, name, value) -> None:
        super().__init__(
            f"Attribute of the wdoc env instance should not be set manually, instead modify os.environ. Attribute name was '{name}'. Value was '{value}'"
        )


class MissingDocdictArguments(Exception):
    """
    Raised when a document loader is called with the wrong number of arguments
    or missing required arguments.
    """

    def __init__(
        self, message: str = "Document loader called with missing arguments"
    ) -> None:
        super().__init__(message)


class NoInferrableFiletype(Exception):
    """
    Occurs when the 'filetype' argument of a file
    was left by the user to 'auto' but wdoc failed to
    find the appropriate loader for it.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
