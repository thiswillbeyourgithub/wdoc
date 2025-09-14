"""
Shared utilities for query and search tasks.
"""

import copy
from beartype.typing import Tuple

from wdoc.utils.misc import log_and_time_fn


@log_and_time_fn
def split_query_parts(query: str) -> Tuple[str, str]:
    """
    Split query into parts for embedding search and answering.
    
    If the query contains ">>>>", splits it into:
    - query_for_embedding: part before >>>>
    - query_to_answer: part after >>>>
    
    Otherwise returns the same query for both purposes.
    
    Parameters
    ----------
    query : str
        The input query string
        
    Returns
    -------
    Tuple[str, str]
        A tuple of (query_for_embedding, query_to_answer)
        
    Raises
    ------
    AssertionError
        If query contains more than one occurrence of ">>>>"
    """
    if ">>>>" in query:
        sp = query.split(">>>>")
        assert (
            len(sp) == 2
        ), "The query must contain a maximum of 1 occurence of '>>>>'"
        query_fe = sp[0].strip()
        query_an = sp[1].strip()
    else:
        query_fe, query_an = copy.copy(query), copy.copy(query)
    
    return query_fe, query_an
