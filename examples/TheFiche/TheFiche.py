"""
TheFiche: A module for generating and exporting Logseq pages based on wdoc queries.

This module provides functionality to create structured Logseq pages
with content generated from wdoc queries, including metadata and properties.
"""

import json
import inspect
from LogseqMarkdownParser import LogseqBlock, LogseqPage, parse_file, parse_text
from typing import List, Dict
import time
from textwrap import indent
from wdoc import wdoc
import litellm
import fire
from pathlib import Path, PosixPath
from datetime import datetime
from beartype import beartype
from typing import Union, Tuple, Optional, Callable
from loguru import logger
from joblib import Memory
import re

VERSION = "1.2"

# logger
logger.add(
    "logs.txt",
    rotation="100MB",
    retention=5,
    format='{time} {level} {thread} TheFiche {process} {function} {line} {message}',
    level="DEBUG",
    enqueue=True,
    colorize=False,
)
def p(*args):
    "simple way to log to logs.txt and to print"
    print(*args)
    logger.info(*args)

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

mem = Memory(".cache", verbose=False)

def chat(
    model: str,
    messages: List[Dict],
    temperature: float,
    **kwargs: Dict,
    ) -> Dict:
    """call to an LLM api. Cached."""
    answer = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False,
        **kwargs,
    ).json()
    assert all(a["finish_reason"] == "stop" for a in answer["choices"]), f"Found bad finish_reason: '{answer}'"
    return answer


def run_wdoc(query: str, kwargs2: dict) -> Tuple[wdoc, dict]:
    "call to wdoc, optionaly cached"
    instance = wdoc(
        task="query",
        import_mode=True,
        query=query,
        **kwargs2,
    )
    fiche = instance.query_task(query=query)
    assert "error" not in fiche

    if len(fiche["all_intermediate_answers"]) > 1:
        extra = '->'.join(
            [str(len(ia)) for ia in fiche["all_intermediate_answers"]]
        )
        extra = f"({extra})"
    else:
        extra = ""

    props = {
        "block_type": "wdoc_the_fiche",
        "wdoc_version": instance.VERSION,
        "wdoc_model": f"{instance.modelname} of {instance.modelbackend}",
        "wdoc_evalmodel": f"{instance.query_eval_modelname} of {instance.query_eval_modelbackend}",
        "wdoc_evalmodel_check_nb": instance.query_eval_check_number,
        "wdoc_cost": f"{float(instance.latest_cost):.5f}",
        "wdoc_n_docs_found": len(fiche["unfiltered_docs"]),
        "wdoc_n_docs_filtered": len(fiche["filtered_docs"]),
        "wdoc_n_docs_used": len(fiche["relevant_filtered_docs"]),
        "wdoc_n_combine_steps": str(len(fiche["all_intermediate_answers"])) + " " + extra,
        "wdoc_kwargs": json.dumps(kwargs2, ensure_ascii=False),
        "the_fiche_version": VERSION,
        "the_fiche_date": today,
        "the_fiche_timestamp": int(time.time()),
        "the_fiche_query": query,
    }
    return fiche, props

@beartype
class TheFiche:
    """
    A class for generating and exporting Logseq pages based on wdoc queries.

    This class encapsulates the process of creating a Logseq page with content
    generated from a wdoc query, including metadata and properties.
    """
    def __init__(
        self,
        query: str,
        logseq_page: Union[str, PosixPath],
        overwrite: bool = False,
        sources_location: str = "as_pages",
        sources_ref_as_prop: bool = False,
        use_cache: bool = True,
        logseq_linkify: bool = True,
        the_fiche_callback: Optional[Callable] = None,
        **kwargs,
        ):
        """
        Initialize a TheFiche instance and generate a Logseq page.

        Args:
            query (str): The query to be processed by wdoc.
            logseq_page (Union[str, PosixPath]): The path to the Logseq page file.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False. If False, will append to the file instead of overwriting. Else, will also overwrite sources if present.
            sources_location (str): If 'as_pages', will store each source as its own page in a 'TheFiche___' namespace. If 'below', sources will be written at the end of the page. Default to "as_pages".
            sources_ref_as_prop (bool): if True, make sure the sources appear as block properties instead of leaving them as is. Default to False.
            use_cache (bool): set to False to bypass the cache, default True.
            logseq_linkify (bool): If True, will ask wdoc's strong LLM to find the import keywords and automatically replace them in the output file by logseq [[links]], enabling the use of graph properties. Default to True.
            the_fiche_callback (callable): will be called at the end of the run with as argument all what's in locals(). Use only if you know what you're doing.
            **kwargs: Additional keyword arguments to pass to wdoc.

        Raises:
            AssertionError: If the file exists and overwrite is False, or if the ratio of used/found documents is too high.
        """
        p(f"Starting TheFiche with args: {query}, {logseq_page}, {overwrite}, {sources_location} and kwargs: {kwargs}")
        assert sources_location in ["as_pages", "below"]
        logseq_page = Path(logseq_page)

        all_kwargs = kwargs.copy()

        # posixpath can't be serialized to json
        for k, v in all_kwargs.items():
            if isinstance(v, PosixPath):
                all_kwargs[k] = str(v)

        if use_cache:
            cached = mem.cache(run_wdoc)
        else:
            cached = run_wdoc

        fiche, props = cached(query=query, kwargs2=all_kwargs)

        p(f"Fiche properties: {props}")

        n_used = props["wdoc_n_docs_used"]
        n_found = props["wdoc_n_docs_found"]

        text = fiche["final_answer"]
        assert text.strip()

        # prepare sources hash dict
        doc_hash = {
            d.metadata["content_hash"][:5]: d.metadata
            for d in fiche["filtered_docs"]
        }
        # discard sources that are not used
        used_hash = []
        for d, dm in doc_hash.items():
            if d in text:
                used_hash.append(d)
            # note: we replace content_hash by all_hash to avoid collisions, as sources are originally refered only by the first 5 letters of content_hash, we now use 5 of all_hash
            new_h = dm["all_hash"][:5]
            text = text.replace(d, f" [[{new_h}]] ")
        text = text.replace(" , ", ", ")
        if not used_hash:
            p("No documents seem to be sourced")

        if logseq_linkify:
            if use_cache:
                cached_chat =  mem.cache(chat)
            else:
                cached_chat = chat
            litellm.drop_params = True
            # remove the sources from the text
            text_wo_s = text
            for dh, dm in doc_hash.items():
                new_h = dm["all_hash"][:5]
                assert dh not in text_wo_s
                if new_h in text_wo_s:
                    assert f"[[{new_h}]]" in text_wo_s
                    text_wo_s = re.sub(
                        f"[, ]*\[\[{new_h}\]\][, ]*",
                        "",
                        text_wo_s,
                    )
                    assert f"[[{new_h}]]" not in text_wo_s
                    assert new_h not in text_wo_s
                    text_wo_s = text_wo_s.replace(" []", "")
                    text_wo_s = text_wo_s.replace("[]", "")
            messages = [
                {
                    "role": "system",
                    "content": """Your task is, given a text, to reply a newline separated list of tags while following some rules.
The purpose of this task is to identify the strings in my text that I could use to create graph like links in my corpus of texts.

Rules:
'''
- The tags can contain multiple words.
- The tags have to be exclusive, for example "recette de cuisine" is bad because you can use instead "cuisine".
- The tags have to be as atomic as possible, for example instead of "HER2 gene" use "HER2".
- Use the singular form of the tag, for exampel not "ulcères" but "ulcère".
- The tags ALWAYS have to exist in the text as is. This is the most important rule. You have to respect the accents, the letters used, etc. Even if it's misspelled!
- The tags have to be about details and specifics, not really big picture stuff. Think for example which drugs are mentionned in this lesson.
- Write each keyword in its own line, so newline separated.
- Take a deep breath before answering.
- Wrap your tags list in a <tags> banner.
- Use as many tags as you deem fit. An absolute minimum of 1 every 3 lines seems about right.
- I won't accept a tag that is not a substring of the whole text. IT HAS TO APPEAR IN THE TEXT AS IS.
'''


Example of an appropriate answer on a text about hemostasis:
'''
Okay, I've read the whole text. I'm now taking a deep breath.
.
.
Okay here's my answer.

<tags>
hemostase
ADAMTS13
VWF
facteur von willebrand
microangiopathie thrombotique
PTT
purpura thrombotique thrombopenique
purpura
SHU
syndrome hemolytique et uremique
infection
anticorps
</tags>
'''
"""
                },
                {"role": "user", "content": text_wo_s},
            ]
            tag_answer = cached_chat(
                messages=messages,
                model=inspect.signature(wdoc).parameters["modelname"].default if "modelname" not in kwargs else kwargs["modelname"],
                temperature=0,
            )
            tags_text = tag_answer["choices"][0]["message"]["content"]
            assert "<tags>" in tags_text and "</tags>" in tags_text, f"missing <tags> or </tags> in new text:\n'''\n{tags_text}\n'''"
            tags = tags_text.split("<tags>", 1)[1].split("</tags>", 1)[0].splitlines()
            tags = [t.strip() for t in tags if t.strip()]
            for t in tags:
                assert "[[" not in t and "]]" not in t, f"tag contains brackets: {t}"
            tags = sorted(tags, key=len)
            assert tags, f"No tags found in tags_text:\n'''\n{tags_text}\n'''"
            text_wo_tags = text

            # tags can be part of another tag
            def find_substrings(string_list: List[str]) -> List[str]:
                result = []
                for i, s1 in enumerate(string_list):
                    for j, s2 in enumerate(string_list):
                        if i != j and s1 in s2:
                            result.append(s1)
                            break
                result = list(set(result))
                assert len(result) <= len(string_list)
                return result
            sub_tags = find_substrings(tags)
            sup_tags = [t for t in tags if t not in sub_tags and re.search(t, text, re.IGNORECASE)]
            if sup_tags:
                for it, t in enumerate(sup_tags):
                    tb = "\b" + t + "\b"
                    if not re.search(tb, text, re.IGNORECASE):
                        p(f"Couldn't find tag {tb} in text")
                    else:
                        p(f"Found tag {tb} in text")
                        text = re.sub(tb, "[[" + t + "]]", text, re.IGNORECASE)
            if sub_tags:
                assert sup_tags
                lines = text.splitlines(keepends=True)
                new_text = ""
                for line in lines:
                    for t in sub_tags:
                        t = t.strip()
                        assert t
                        assert "[" not in t and "]" not in t
                        tb = "\b" + t + "\b"
                        if not re.search(tb, line, re.IGNORECASE):
                            continue

                        if line == line.rstrip():
                            line = line + " [[" + t + "]]"
                        else:
                            head = line.rstrip()
                            tail = line[len(head):]
                            assert head == head.rstrip()
                            assert tail != tail.rstrip(), tail
                            line = head + " [[" + t + "]]" + tail
                            assert line != line.rstrip()

                    new_text += line
                # assert text != new_text
                if text == new_text:
                    p(f"Text was not different after checking for subtags, which are: '{sub_tags}'")

                text = new_text

        # create logseq page object but don't save it yet
        if not text.startswith("- "):
            text = "- " + text
        # fix logseq indentation
        if True:  # or text.count("    ") % text.count("  ") != 0:
            text = text.replace("  ", "    ")
        content = parse_text(text)
        content.page_properties.update(props)
        assert content.page_properties

        # add documents source
        if sources_location == "below":
            content.blocks.append(LogseqBlock("- # Sources\n  collapsed:: true"))
            for dh, dm in doc_hash.items():
                if dh not in used_hash:
                    continue
                cont = [
                    fd.page_content for fd in fiche["filtered_docs"]
                    if fd.metadata["all_hash"] == dm["all_hash"]
                ]
                assert len(cont) == 1, f"Found multiple sources with the same hash! {cont}"
                cont = cont[0].strip()
                new_block = LogseqBlock(f"- [[{dh}]]: {indent(cont, '  ').strip()}")
                for k, v in dm.items():
                    new_block.set_property(k, v)
                diff = (4 - new_block.indentation_level % 4)
                if diff != 0:
                    diff += 4
                new_block.indentation_level += 4 + diff
                content.blocks.append(new_block)

        elif sources_location == "as_pages":
            logseq_dir = logseq_page.parent
            for dh, dm in doc_hash.items():
                if dh not in used_hash:
                    continue
                source_path = logseq_dir / ("TheFicheSources___" + dm["all_hash"] + ".md")
                cont = [
                    fd.page_content for fd in fiche["filtered_docs"]
                    if fd.metadata["all_hash"] == dm["all_hash"]
                ]
                assert len(cont) == 1, f"Found multiple sources with the same hash! {cont}"
                cont = cont[0].strip()
                new_h = dm["all_hash"][:5]

                if overwrite:
                    source_path.unlink(missing_ok=True)

                conflict = False
                if source_path.exists():
                    p(f"Warning: a source for {dh} ({new_h}) already exists at {source_path}")
                    prev_source = parse_file(source_path)
                    if prev_source.page_properties["content_hash"][:5].startswith(dh):
                        if not prev_source.page_properties["all_hash"].startswith(new_h):
                            p(f"Found previous source with the same name and overlapping content but different alias: page: {source_path}, dh: {dh}, new_h: {new_h}\nCreating a conflict file")
                            conflict = True
                    else:
                        p(f"Found previous source with the same name but does not contain the new content: {source_path}, dh: {dh}, new_h: {new_h}\nCreating a conflict file")
                        conflict = True

                if conflict:
                    sp = source_path.parent / (source_path.name + "_conflict")
                else:
                    sp = source_path
                logger.info(f"Creating source page for {dh} ({new_h}) at {sp}")
                cont = indent(cont, "  ").strip()
                if not cont.startswith("- "):
                    cont = "- " + cont
                source_page = parse_text(cont)
                source_page.page_properties.update(dm.copy())
                source_page.page_properties["alias"] = new_h
                source_page.export_to(
                    file_path=sp.absolute(),
                    overwrite=True,
                    allow_empty=False,
                )
        else:
            raise ValueError(sources_location)

        if sources_ref_as_prop:
            # make it so that the sources appear as block properties instead of in the content
            for ib, b in enumerate(content.blocks):
                for dh, dm in doc_hash.items():
                    new_h = dm["all_hash"][:5]
                    assert dh not in b.content
                    if new_h in b.content:
                        assert f"[[{new_h}]]" in b.content
                        content.blocks[ib].content = re.sub(
                            f"[, ]*\[\[{new_h}\]\][, ]*",
                            "",
                            b.content,
                        )
                        assert f"[[{new_h}]]" not in b.content
                        assert new_h not in content.blocks[ib].content
                        content.blocks[ib].content = content.blocks[ib].content.replace(" []", "")
                        content.blocks[ib].content = content.blocks[ib].content.replace("[]", "")
                        content.blocks[ib].set_property("source", f"[[{new_h}]]")


        # save to file
        if (not logseq_page.absolute().exists()) or overwrite:
            content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=False if not overwrite else True,
                allow_empty=False,
            )
        else:
            prev_content = parse_file(logseq_page)
            prev_content.blocks.append(LogseqBlock("- ---"))

            new_block = LogseqBlock(f"- # {today}")
            for k, v in content.page_properties.items():
                new_block.set_property(k, v)

            prev_content.blocks.append(new_block)
            # to over indent the new blocks
            # for block in content.blocks:
            #     diff = (4 - block.indentation_level % 4)
            #     if diff != 0:
            #         diff += 4
            #     block.indentation_level += 4 + diff
            #     prev_content.blocks.append(block)
            prev_content.export_to(
                file_path=logseq_page.absolute(),
                overwrite=True,
                allow_empty=False,
            )

        if the_fiche_callback:
            p(f"Using callback '{the_fiche_callback}'")
            the_fiche_callback(**locals())

        p("Done")


if __name__ == "__main__":
    fire.Fire(TheFiche)
