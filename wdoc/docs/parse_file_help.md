
# 'parse_file' documentation

## Description

Simple function to load a document given at least  path arg. Used for cli
and convenience in python scripts.

## Arguments

- `path`: str
    - Same as for wdoc (can be None, for example if filetype is `anki`).

- `filetype`: str
    - Same as for wdoc

- `format`: str, default `text`
    - if `text`: only return the text
    - if `xml`: returns text in an xml like format
    - if `langchain`: return a list of langchain Documents
    - if `langchain_dict`: return a list of langchain Documents as
        python dicts (easy to json parse, and metadata are included)

- `cli_kwargs`: dict, default `None`
    - Dict containing keyword arguments destined to the function
    `batch_load_doc` and not about a specific document per say.
    e.g. "file_loader_n_jobs", etc.

- `debug`: bool, default `False`
    - Same as for wdoc

- `verbose`: bool, default `False`
    - Same as for wdoc

- `**kwargs`
    - Remaning keyword arguments are assumed to be DocDict arguments,
    the full list is at wdoc.utils.misc.filetype_arg_types

## Return value
- Either the document's page_content as a string, or a list of
langchain Document (so with attributes `page_content` and `metadata`).
