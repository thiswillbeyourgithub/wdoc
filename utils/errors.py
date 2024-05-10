class NoDocumentsRetrieved(Exception):
    def __init__(self, message: str="no documents were retrieved.") -> None:
        super().__init__(message)

class NoDocumentsAfterWeakLLMFiltering(Exception):
    def __init__(self, message: str="No document remained after filtering with the query") -> None:
        super().__init__(message)

