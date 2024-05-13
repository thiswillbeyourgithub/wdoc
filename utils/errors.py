class NoDocumentsRetrieved(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class NoDocumentsAfterWeakLLMFiltering(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class InvalidDocEvaluationByWeakLLM(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

