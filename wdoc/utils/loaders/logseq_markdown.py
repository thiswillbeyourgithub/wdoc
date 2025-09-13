import copy
from pathlib import Path

import LogseqMarkdownParser
from beartype.typing import List, Union
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from loguru import logger

from wdoc.utils.loaders.shared import debug_return_empty, markdownimage_regex
from wdoc.utils.misc import doc_loaders_cache, optional_strip_unexp_args


@debug_return_empty
@optional_strip_unexp_args
@doc_loaders_cache.cache(ignore=["path"])
def load_logseq_markdown(
    path: Union[str, Path],
    file_hash: str,
    text_splitter: TextSplitter,
) -> List[Document]:
    path = Path(path)
    logger.info(f"Loading logseq markdown file: '{path}'")
    assert path.exists(), f"file not found: '{path}'"
    try:
        parsed = LogseqMarkdownParser.parse_file(path, verbose=False)
    except Exception as err:
        raise Exception(f"Error when parsing {path} LogseqMarkdownParser: '{err}'")

    if not parsed.blocks:
        raise Exception(
            f"No logseq blocks loaded for {path} (file size: {Path(path).stat().st_size})"
        )

    blocks = parsed.blocks
    page_props = parsed.page_properties

    content = parsed.content
    content = content.replace("\t", "    ")
    content = markdownimage_regex.sub("[IMAGE]", content)
    # content, _ = replace_media(
    #     content=content,
    #     media=None,
    #     mode="remove_media",
    #     strict=False,
    #     replace_image=True,
    #     replace_links=True,
    #     replace_sounds=False,
    # )

    # create a single document then for each document add the properties of each block found in the doc
    docs = text_splitter.transform_documents(
        [
            Document(
                page_content=content,
                metadata=page_props,
            )
        ]
    )

    failed_blocks = []
    for b in blocks:
        b = copy.copy(b)
        props = b.properties.copy()
        for k, v in props.items():
            b.del_property(key=k)
            b.content = b.content.strip()
        cont = b.content.replace("\t", "    ")
        cont = markdownimage_regex.sub("[IMAGE]", cont)
        # cont, _ = replace_media(
        #     content=cont,
        #     media=None,
        #     mode="remove_media",
        #     strict=False,
        #     replace_image=True,
        #     replace_links=True,
        #     replace_sounds=False,
        # )
        if not cont:
            continue
        found = False
        for i, d in enumerate(docs):
            if i + 1 >= len(docs):
                next = ""
            else:
                next = docs[i + 1].page_content
            if cont.strip() in d.page_content or (
                cont not in next and cont in d.page_content + next
            ):

                # merge metadata dictionnaries
                for k, v in props.items():
                    if not v:
                        continue
                    if k not in docs[i].metadata:
                        docs[i].metadata[k] = v
                    elif docs[i].metadata[k] == v:
                        continue
                    elif isinstance(docs[i].metadata[k], list):
                        if isinstance(v, list):
                            docs[i].metadata[k].extend(v)
                        else:
                            docs[i].metadata[k].append(v)
                    else:
                        assert k in docs[i].metadata
                        assert not isinstance(docs[i].metadata[k], list)
                        assert docs[i].metadata[k] != v
                        if isinstance(v, list):
                            docs[i].metadata[k] = [docs[i].metadata[k]] + v
                        else:
                            docs[i].metadata[k] = [docs[i].metadata[k], v]
                found = True
                break
        if not found:
            failed_blocks.append(b)

    if failed_blocks:
        mess = f"Couldn't find {len(failed_blocks)} block(s) out of {len(blocks)} after splitting the logseq page."
        mess += "\nBlocks were:"
        for b in failed_blocks:
            mess += "\n" + str(b)
        if len(failed_blocks) >= 0.5 * len(blocks):
            mess += "\nMissing more than 50% of blocks so crashing"
            raise Exception(mess)
        else:
            logger.warning(mess + "\nBut continuing nonetheless")

    # sort and deduplicate metadata
    for i, d in enumerate(docs):
        for k, v in d.metadata.items():
            if isinstance(v, list):
                d.metadata[k] = list(sorted(list(set(v))))
            assert d.metadata[
                k
            ], f"There shouldn't be any empty metadata value but key '{k}' of doc '{d}' is empty."

    return docs
