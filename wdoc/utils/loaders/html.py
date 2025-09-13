from pathlib import Path

import bs4
import dill
from beartype.typing import Callable, List, Optional, Union
from langchain.docstore.document import Document
from loguru import logger

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import doc_loaders_cache, optional_strip_unexp_args


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_local_html(
    path: Union[str, Path],
    file_hash: str,
    load_functions: Optional[bytes] = None,
) -> List[Document]:
    path = Path(path)
    logger.info(f"Loading local html: '{path}'")
    assert path.exists(), f"file not found: '{path}'"

    content = path.read_text()

    if load_functions:
        # the functions must be pickled because joblib can't
        # cache string that would declare as lambda functions

        try:
            load_functions = dill.loads(load_functions)
        except Exception as err:
            raise Exception(f"Error when unpickling load_functions: '{err}'")
        assert isinstance(
            load_functions, tuple
        ), f"load_functions must be a tuple, not {type(load_functions)}"
        assert all(
            callable(lf) for lf in load_functions
        ), f"load_functions element must be a callable, not {[type(lf) for lf in load_functions]}"

        for ifunc, func in enumerate(load_functions):
            try:
                content = func(content)
            except Exception as err:
                raise Exception(
                    f"load_functions #{ifunc}: '{func}' failed with " f"error : '{err}'"
                )
        assert isinstance(content, str), (
            f"output of function #{ifunc}: '{func}' is not a " f"string: {content}"
        )
    try:
        soup = bs4.BeautifulSoup(content, "html.parser")
    except Exception as err:
        raise Exception(f"Error when parsing html: {err}")

    text = soup.get_text().strip()
    assert text, "Empty text after loading from html"

    docs = [
        Document(
            page_content=text,
        )
    ]
    return docs


@doc_loaders_cache.cache
def eval_load_functions(
    load_functions: str,
) -> List[Callable]:
    assert isinstance(load_functions, list), "load_functions must be of type list"
    assert all(
        isinstance(lf, str) for lf in load_functions
    ), "elements of load_functions must be of type str"

    try:
        for ilf, lf in enumerate(load_functions):
            load_functions[ilf] = eval(lf)
    except Exception as err:
        raise Exception(f"Error when evaluating load_functions #{ilf}: {lf} '{err}'")
    assert all(
        callable(lf) for lf in load_functions
    ), f"Some load_functions are not callable: {load_functions}"
