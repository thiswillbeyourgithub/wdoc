import json
from pathlib import Path

from beartype.typing import List, Optional, Union
from langchain.docstore.document import Document

from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import doc_loaders_cache, optional_strip_unexp_args


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_json_dict(
    path: Union[str, Path],
    json_dict_template: str,
    file_hash: str,
    metadata: Optional[Union[str, dict]] = None,
    json_dict_exclude_keys: Optional[List[str]] = None,
) -> List[Document]:
    path = Path(path)
    assert path.exists(), f"file not found: '{path}'"

    assert "{key}" in json_dict_template, "json_dict_template must contain '{key}'"
    assert "{value}" in json_dict_template, "json_dict_template must contain '{value}'"

    with Path(path).open("r") as f:
        d = json.load(f)
    assert d, "dict is empty"

    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not metadata:
        metadata = {}
    if json_dict_exclude_keys is None:
        json_dict_exclude_keys = []

    docs = []
    for k, v in d.items():
        if k in json_dict_exclude_keys:
            continue
        doc = Document(
            page_content=json_dict_template.replace("{key}", k).replace("{value}", v),
            metadata=metadata,
        )
        docs.append(doc)
    assert docs, "No document found in json_dict"

    return docs
