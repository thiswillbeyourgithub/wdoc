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
    def __init__(self, name, value) -> None:
        super().__init__(
            f"Attribute of the wdoc env instance should not be set manually, instead modify os.environ. Attribute name was '{name}'. Value was '{value}'"
        )
