# Parse Doc

## Description

`parse_doc` is the function called when you do `wdoc parse_doc --path=my_path`.
It takes as argument basically the file related arguments of wdoc and completely
bypasses anything related to summarising, querying, LLM etc. Hence it is meant
to be used as an utility that parses any input to text. You can for example
use it to quickly parse anything to send to [@simonw's](https://github.com/simonw/) [llm](https://github.com/simonw/llm) or any other .shell utility.

## Arguments

- `filetype`: str
    - Same as for wdoc

- `format`: str, default `text`
    - if `text`: only return the text
    - if `xml`: returns text in an xml like format
    - if `langchain`: return a list of langchain Documents
    - if `langchain_dict`: return a list of langchain Documents as
        python dicts (easy to json parse, and metadata are included)

- `debug`: bool, default `False`
    - Same as for wdoc

- `verbose`: bool, default `False`
    - Same as for wdoc

- `out_file`: str or Path, default `None`
    - If specified, writes the output to the given file path.
    - If the file exists and is binary, the function will crash.
    - Otherwise, the output will be appended to the file (no overwrite).
    - The output is still returned normally for programmatic use.

- `**kwargs`
    - Remaning keyword arguments are assumed to be DocDict arguments,
    the full list is at wdoc.utils.misc.filetype_arg_types
    or in the "DocDict arguments" section of `wdoc --help`.

## Return value
- Either the document's page_content as a string, or a list of
langchain Document (so with attributes `page_content` and `metadata`).
