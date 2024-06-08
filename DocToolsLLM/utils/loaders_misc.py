from .typechecker import optional_typecheck
from .logger import red

from typing import List

import tiktoken
import fire

import lazy_import
Document = lazy_import.lazy_class('langchain.docstore.document.Document')
TextSplitter = lazy_import.lazy_class('langchain.text_splitter.TextSplitter')
RecursiveCharacterTextSplitter = lazy_import.lazy_class('langchain.text_splitter.RecursiveCharacterTextSplitter')

# parse args again to know what to print for failed imports
kwargs = fire.Fire(lambda *args, **kwargs: kwargs)
if "debug" in kwargs and kwargs["debug"]:
    verbose = True
else:
    verbose = False
try:
    ftlangdetect = lazy_import.lazy_module("ftlangdetect")
except Exception as err:
    if verbose:
        print(f"Couldn't import optional package 'ftlangdetect', trying to import langdetect (but it's much slower): '{err}'")
    try:
        import langdetect
    except Exception as err:
        if verbose:
            print(f"Couldn't import optional package 'langdetect': '{err}'")

if "ftlangdetect" in globals():
    @optional_typecheck
    def language_detector(text: str) -> float:
        return ftlangdetect.detect(text)["score"]
elif "language_detect" in globals():
    @optional_typecheck
    def language_detector(text: str) -> float:
        return langdetect.detect_langs(text)[0].prob
else:
    def language_detector(text: str) -> None:
        return None


# for reading length estimation
wpm = 250
average_word_length = 6

# separators used for the text splitter
recur_separator = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "...", ".", " ", ""]

tokenize = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
).encode  # used to get token length estimation

min_token = 50
max_token = 1_000_000
max_lines = 100_000
min_lang_prob = 0.50


@optional_typecheck
def get_tkn_length(tosplit: str) -> int:
    return len(tokenize(tosplit))

@optional_typecheck
def get_splitter(task: str) -> TextSplitter:
    "we don't use the same text splitter depending on the task"
    if task in ["query", "search"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=3000,  # default 4000
            chunk_overlap=386,  # default 200
            length_function=get_tkn_length,
        )
    elif task in ["summarize_then_query", "summarize"]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=2000,
            chunk_overlap=300,
            length_function=get_tkn_length,
        )
    elif task == "recursive_summary":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=recur_separator,
            chunk_size=1000,
            chunk_overlap=200,
            length_function=get_tkn_length,
        )
    else:
        raise Exception(task)
    return text_splitter


@optional_typecheck
def check_docs_tkn_length(docs: List[Document], name: str) -> float:
    """checks that the number of tokens in the document is high enough,
    not too low, and has a high enough language probability,
    otherwise something probably went wrong."""
    size = sum([get_tkn_length(d.page_content) for d in docs])
    nline = len("\n".join([d.page_content for d in docs]).splitlines())
    if nline > max_lines:
        red(
            f"Example of page from document with too many lines : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of lines from '{name}' is {nline} > {max_lines}, probably something went wrong?"
        )
    if size <= min_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} <= {min_token}, probably something went wrong?"
        )
    if size >= max_token:
        red(
            f"Example of page from document with too many tokens : {docs[len(docs)//2].page_content}"
        )
        raise Exception(
            f"The number of token from '{name}' is {size} >= {max_token}, probably something went wrong?"
        )

    # check if language check is above a threshold
    prob = [language_detector(docs[0].page_content.replace("\n", "<br>"))]
    if prob[0] is None:
        # bypass if language_detector not defined
        return 1
    if len(docs) > 1:
        prob.append(language_detector(docs[1].page_content.replace("\n",
                                "<br>")))
        if len(docs) > 2:
            prob.append(
                    language_detector(
                        docs[len(docs) // 2].page_content.replace("\n", "<br>")
                    )
            )
    prob = max(prob)
    if prob <= min_lang_prob:
        red(
            f"Low language probability for {name}: prob={prob:.3f}<{min_lang_prob}.\nExample page: {docs[len(docs)//2]}"
        )
        raise Exception(
            f"Low language probability for {name}: prob={prob:.3f}.\nExample page: {docs[len(docs)//2]}"
        )
    return prob

