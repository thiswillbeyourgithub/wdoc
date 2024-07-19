"""
Exception classes
"""


class NoDocumentsRetrieved(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class NoDocumentsAfterLLMEvalFiltering(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidDocEvaluationByLLMEval(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
