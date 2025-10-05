"""
Chain (logic) used to summarize a document.
"""

from datetime import date
from pathlib import Path
import re
from dataclasses import MISSING
import json
from beartype.typing import List, Tuple, Dict, Optional, Union
from langchain.docstore.document import Document
from tqdm import tqdm
from loguru import logger
import copy
from dataclasses import dataclass, asdict

from wdoc.utils.logger import (
    md_printer,
)
from wdoc.utils.misc import (
    # debug_chain,
    ModelName,
    average_word_length,
    check_docs_tkn_length,
    get_splitter,
    get_tkn_length,
    log_and_time_fn,
    thinking_answer_parser,
    wpm,
)
from wdoc.utils.prompts import (
    BASE_SUMMARY_PROMPT,
    PREV_SUMMARY_TEMPLATE,
    RECURSION_INSTRUCTION,
)
from wdoc.utils.env import env

HOME = str(Path.home())


@dataclass
class wdocSummary:
    """
    Container for document summarization results with dict-like access.

    This dataclass encapsulates all outputs from the document summarization process,
    including metrics, costs, and the summary text itself. It provides dict-like
    access for backward compatibility while offering better type safety and cleaner
    code structure.

    Attributes
    ----------
    path : str
        Original document path or URL that was summarized.
    summary : str
        Final summary text from the best recursion pass.
    recursive_summaries : Dict[int, str]
        Mapping of recursion level to summary text for each pass.
    sum_reading_length : float
        Estimated reading time in minutes for the final summary.
    sum_tkn_length : int
        Token count of the final summary text.
    doc_reading_length : float
        Original document reading time in minutes.
    doc_total_tokens : Dict[str, int]
        Token usage breakdown by type (prompt, completion, internal_reasoning).
    doc_total_tokens_sum : int
        Total tokens used across all operations.
    doc_total_tokens_str : str
        Human-readable string representation of token usage.
    doc_total_cost : Union[float, int]
        Total cost in dollars for LLM usage.
    author : Optional[str]
        Document author if available in metadata.
    n_chunk : int
        Number of document chunks that were processed.
    """

    path: str
    summary: str
    recursive_summaries: Dict[int, str]
    sum_reading_length: float
    sum_tkn_length: int
    doc_reading_length: float
    doc_total_tokens: Dict[str, int]
    doc_total_tokens_sum: int
    doc_total_tokens_str: str
    doc_total_cost: Union[float, int]
    author: Optional[str]
    n_chunk: int

    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow dict-like assignment for backward compatibility."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like behavior."""
        return hasattr(self, key)

    def keys(self):
        """Return field names like dict.keys()."""
        return asdict(self).keys()

    def values(self):
        """Return field values like dict.values()."""
        return asdict(self).values()

    def items(self):
        """Return field name-value pairs like dict.items()."""
        return asdict(self).items()

    def get(self, key: str, default=None):
        """Get value with default like dict.get()."""
        return getattr(self, key, default)


def summarize_documents(
    path: Union[str, Path],
    relevant_docs: List,
    summary_language: str,
    model: ModelName,
    llm: Union[
        "langchain_litellm.ChatLiteLLM",
        "langchain_community.chat_models.fake.FakeListChatModel",
    ],
    llm_verbosity: bool,
    summary_n_recursion: int,
    llm_price: dict,
    in_import_mode: bool,
    out_file: Optional[str],
    wdoc_version: str,
) -> wdocSummary:
    """
    Orchestrate the complete document summarization process with optional recursion.

    This function serves as the main entry point for document summarization. It
    extracts metadata from documents, performs initial summarization, optionally
    applies recursive summarization to condense the output further, calculates
    costs and reading times, and handles output formatting and file writing.
    Recursive summarization continues until the summary converges or reaches
    the specified recursion limit.

    Parameters
    ----------
    path : Union[str, Path]
        Source path or URL of the document being summarized. Used for metadata
        and identification purposes.
    relevant_docs : List
        List of Document objects containing the content to summarize. Must not
        be empty and should contain metadata like 'title', 'author', etc.
    summary_language : str
        Target language for the summary output.
    model : ModelName
        Model configuration object containing backend and tokenization info.
    llm : Union[langchain_litellm.ChatLiteLLM, langchain_community.chat_models.fake.FakeListChatModel]
        Language model instance for generating summaries.
    llm_verbosity : bool
        If True, enables verbose logging of LLM interactions and intermediate outputs.
    summary_n_recursion : int
        Maximum number of recursive summarization passes. 0 means no recursion.
        Each pass attempts to further condense the previous summary.
    llm_price : dict
        Pricing information for token usage calculation with keys matching
        token types ('prompt', 'completion', 'internal_reasoning').
    in_import_mode : bool
        If True, suppresses console output for integration scenarios.
    out_file : Optional[str]
        Path to output file for saving the summary. If None, no file is written.
        Intermediate recursion summaries are saved with numbered extensions.
    wdoc_version : str
        Version string of wdoc for metadata tracking.

    Returns
    -------
    wdocSummary
        Comprehensive summary results containing all metrics, costs, and summary text.
        Can be accessed as a dict for backward compatibility.

    Raises
    ------
    AssertionError
        If relevant_docs is empty or contains unexpected data.

    Notes
    -----
    Recursive summarization stops early if:
    - The summary text becomes identical to the previous iteration
    - Token length validation fails for the recursive summary chunks
    - Maximum recursion depth is reached

    The function prioritizes cost transparency by detailed token tracking and
    supports both interactive and programmatic usage modes through the
    in_import_mode parameter.
    """
    import tldextract

    assert relevant_docs, "Empty relevant_docs!"

    # parse metadata from the doc
    metadata = []
    if "http" in path:
        item_name = tldextract.extract(path).registered_domain
    elif "/" in path and Path(path).exists():
        item_name = Path(path).name
    else:
        item_name = path

    if "title" in relevant_docs[0].metadata:
        item_name = f"{relevant_docs[0].metadata['title'].strip()} - {item_name}"
    else:
        metadata.append(f"<title>\n{item_name.strip()}\n</title>")

    # replace # in title as it would be parsed as a tag
    item_name = item_name.replace("#", r"\#")

    if "doc_reading_time" in relevant_docs[0].metadata:
        doc_reading_length = relevant_docs[0].metadata["doc_reading_time"]
        metadata.append(
            f"<reading_length>\n{doc_reading_length:.1f} minutes\n</reading_length>"
        )
    else:
        doc_reading_length = 0
    if "author" in relevant_docs[0].metadata:
        author = relevant_docs[0].metadata["author"].strip()
        metadata.append(f"<author>\n{author}\n</author>")
    else:
        author = None
    if "yt_chapters" in relevant_docs[0].metadata:
        chapters = json.dumps(relevant_docs[0].metadata["yt_chapters"])
        metadata.append(f"<youtube_chapters>\n{chapters}\n</youtube_chapters>")
    metadata.append(f"<today>\n{date.today().isoformat()}\n</today>")

    if metadata:
        metadata = "<text_metadata>\n" + "\n".join(metadata) + "\n"
        metadata += "<section_number>\n[PROGRESS]\n</section_number>\n"
        metadata += "</text_metadata>"
    else:
        metadata = (
            "<text_metadata><section_number>[PROGRESS]</section_number></text_metadata>"
        )

    # summarize each chunk of the link and return one text
    (
        summary,
        n_chunk,
        doc_total_tokens,
    ) = _summarize(
        docs=relevant_docs,
        metadata=metadata,
        language=summary_language,
        modelbackend=model.backend,
        llm=llm,
        verbose=llm_verbosity,
    )

    # get reading length of the summary
    real_text = "".join([letter for letter in list(summary) if letter.isalpha()])
    sum_reading_length = len(real_text) / average_word_length / wpm
    logger.info(f"{item_name} reading length is {sum_reading_length:.1f}")

    recursive_summaries = {0: summary}
    prev_real_text = MISSING
    if summary_n_recursion > 0:
        for n_recur in range(1, summary_n_recursion + 1):
            summary_text = copy.deepcopy(recursive_summaries[n_recur - 1])
            logger.warning(f"Doing summary check #{n_recur} of {item_name}")

            # remove any chunk count that is not needed to summarize
            sp = summary_text.split("\n")
            for i, l in enumerate(sp):
                if l.strip() == "- ---":
                    sp[i] = None
                elif re.search(r"- Chunk \d+/\d+", l):
                    sp[i] = None
                elif l.strip().startswith("- BEFORE RECURSION #"):
                    for new_i in range(i, len(sp)):
                        sp[new_i] = None
                    break
            summary_text = "\n".join([s.rstrip() for s in sp if s])
            assert "- ---" not in summary_text, "Found chunk separator"
            assert "- Chunk " not in summary_text, "Found chunk marker"
            assert "- BEFORE RECURSION # " not in summary_text, "Found recursion block"

            splitter = get_splitter(
                "recursive_summary",
                modelname=model,
            )
            summary_docs = [Document(page_content=summary_text)]
            summary_docs = splitter.transform_documents(summary_docs)
            assert summary_docs != relevant_docs
            try:
                check_docs_tkn_length(summary_docs, item_name)
            except Exception as err:
                logger.warning(
                    f"Exception when checking if {item_name} could be recursively summarized for the #{n_recur} time: {err}"
                )
                break
            (
                summary_text,
                n_chunk,
                new_doc_total_tokens,
            ) = _summarize(
                docs=summary_docs,
                metadata=metadata,
                language=summary_language,
                modelbackend=model.backend,
                llm=llm,
                verbose=llm_verbosity,
                n_recursion=n_recur,
            )

            # aggregate the token count
            for k, v in new_doc_total_tokens.items():
                doc_total_tokens[k] += v

            # clean text again to compute the reading length
            sp = summary_text.split("\n")
            for i, l in enumerate(sp):
                if l.strip() == "- ---":
                    sp[i] = None
                elif re.search(r"- Chunk \d+/\d+", l):
                    sp[i] = None
                elif l.strip().startswith("- BEFORE RECURSION #"):
                    for new_i in range(i, len(sp)):
                        sp[new_i] = None
                    break
            real_text = "\n".join([s.rstrip() for s in sp if s])
            assert "- ---" not in real_text, "Found chunk separator"
            assert "- Chunk " not in real_text, "Found chunk marker"
            assert "- BEFORE RECURSION # " not in real_text, "Found recursion block"
            real_text = "".join(
                [letter for letter in list(real_text) if letter.isalpha()]
            )
            sum_reading_length = len(real_text) / average_word_length / wpm
            logger.info(
                f"{item_name} reading length after recursion #{n_recur} is {sum_reading_length:.1f}"
            )
            if prev_real_text is not MISSING:
                if real_text == prev_real_text:
                    logger.warning(
                        f"Identical summary after {n_recur} "
                        "recursion, adding more recursion will not "
                        "help so stopping here"
                    )
                    recursive_summaries[n_recur] = summary_text
                    break
            prev_real_text = real_text

            assert n_recur not in recursive_summaries
            if summary_text not in recursive_summaries:
                logger.warning(
                    f"Identical summary after {n_recur} "
                    "recursion, adding more recursion will not "
                    "help so stopping here"
                )
                recursive_summaries[n_recur] = summary_text
                break
            else:
                recursive_summaries[n_recur] = summary_text

    best_sum_i = max(list(recursive_summaries.keys()))
    if not in_import_mode:
        print("\n\n")
        md_printer("# Summary")
        md_printer(f"## {path}")
        md_printer(recursive_summaries[best_sum_i])

    # the price computation needs to happen as late as possible to avoid
    # underflow errors
    doc_total_cost = 0
    doc_total_tokens_str = ""
    for k, v in doc_total_tokens.items():
        if llm_price[k]:  # to avoid underflow errors:
            doc_total_cost += v * llm_price[k]
        doc_total_tokens_str += f"{k.title()}: {v} "
    doc_total_tokens_str = doc_total_tokens_str.strip()
    logger.info(
        f"Tokens used for {path}: ({doc_total_tokens_str}, cost: ${doc_total_cost:.5f})"
    )

    doc_total_tokens_sum = sum(
        [int(number) for number in doc_total_tokens.values() if str(number).isdigit()]
    )

    summary_tkn_length = get_tkn_length(recursive_summaries[best_sum_i])

    header = f"\n- {item_name}    cost: ${doc_total_cost:.5f} ({doc_total_tokens_str})"
    if doc_reading_length:
        header += f"    {doc_reading_length:.1f} minutes"
    if author:
        header += f"    by '{author}'"
    header += f"    original path: '{path}'"
    header += f"    wdoc version {wdoc_version} with model {model} on {date.today().isoformat()}"

    # save to output file
    if out_file:
        if in_import_mode:
            logger.warning(
                "Detected use of out_file arg while in __import_mode__. This is unexpected and might lead to issues."
            )
        for nrecur, summary in recursive_summaries.items():
            out_file = Path(out_file)
            if len(recursive_summaries) > 1 and nrecur < max(
                list(recursive_summaries.keys())
            ):
                # also store intermediate summaries if present
                out_file = out_file.parent / (out_file.stem + f".{nrecur + 1}.md")

            with open(str(out_file), "a") as f:
                if out_file.exists() and out_file.read_text().strip():
                    f.write("\n\n\n")
                f.write(header)
                if len(recursive_summaries) > 1:
                    f.write(
                        f"\n    Recursive summary pass: {nrecur + 1}/{len(recursive_summaries)}"
                    )

                for bulletpoint in summary.split("\n"):
                    f.write("\n")
                    bulletpoint = bulletpoint.rstrip()
                    f.write(f"    {bulletpoint}")

    return wdocSummary(
        path=path,
        summary=recursive_summaries[best_sum_i],
        recursive_summaries=recursive_summaries,
        sum_reading_length=sum_reading_length,
        sum_tkn_length=summary_tkn_length,
        doc_reading_length=doc_reading_length,
        doc_total_tokens=doc_total_tokens,
        doc_total_tokens_sum=doc_total_tokens_sum,
        doc_total_tokens_str=doc_total_tokens_str,
        doc_total_cost=doc_total_cost,
        author=author,
        n_chunk=n_chunk,
    )


@log_and_time_fn
def _summarize(
    docs: List[Document],
    metadata: str,
    language: str,
    modelbackend: str,
    llm: Union[
        "langchain_litellm.ChatLiteLLM",
        "langchain_community.chat_models.fake.FakeListChatModel",
    ],
    verbose: bool,
    n_recursion: int = 0,
) -> Tuple[str, int, Dict[str, int]]:
    """
    Process document chunks through an LLM to generate structured summaries.

    This function iterates through document chunks, sending each to an LLM for
    summarization. It handles progressive context by including previous summaries,
    formats the output as markdown bullet points, and tracks token usage for
    cost calculation. The function ensures consistent formatting by normalizing
    bullet points and handling markdown syntax issues.

    Parameters
    ----------
    docs : List[Document]
        List of document chunks to summarize. Each chunk is processed sequentially.
    metadata : str
        XML-formatted metadata about the document including title, author, progress
        indicators. Must contain "[PROGRESS]" placeholder for chunk tracking.
    language : str
        Target language for the summary output.
    modelbackend : str
        Backend identifier for the LLM model being used.
    llm : Union[langchain_litellm.ChatLiteLLM, langchain_community.chat_models.fake.FakeListChatModel]
        Language model instance for generating summaries. Must support caching.
    verbose : bool
        If True, logs intermediate thinking and summary outputs.
    n_recursion : int, optional
        Current recursion level for recursive summarization, by default 0.
        Adds special instructions when > 0.

    Returns
    -------
    Tuple[str, int, Dict[str, int]]
        - Combined summary text with chunk separators and progress indicators
        - Number of chunks processed
        - Token usage breakdown with keys: 'prompt', 'completion', 'internal_reasoning'

    Notes
    -----
    The function applies several text cleaning operations:
    - Removes LLM artifacts like "take a deep breath" phrases
    - Normalizes bullet points to use consistent "- " format
    - Fixes markdown formatting issues (bold/italic markers)
    - Maintains context between chunks using previous summary snippets
    """
    summaries = []
    previous_summary = ""

    llm.bind(verbose=verbose)

    token_details = {"prompt": 0, "completion": 0, "internal_reasoning": 0}

    metadata = metadata.replace(HOME, "~")  # extra privacy just in case a path appears

    assert "[PROGRESS]" in metadata
    for ird, rd in tqdm(enumerate(docs), desc="Summarising splits", total=len(docs)):
        fixed_index = f"{ird + 1}/{len(docs)}"

        messages = BASE_SUMMARY_PROMPT.format_messages(
            language=language,
            metadata=metadata.replace("[PROGRESS]", fixed_index),
            previous_summary=previous_summary,
            recursion_instruction="" if not n_recursion else RECURSION_INSTRUCTION,
            text=rd.page_content,
        )
        if " object at " in llm._get_llm_string():
            logger.warning(
                "Found llm._get_llm_string() value that potentially "
                f"invalidates the cache: '{llm._get_llm_string()}'\n"
                f"Related github issue: 'https://github.com/langchain-ai/langchain/issues/23257'"
            )
        try:
            output = llm._generate_with_cache(
                messages, request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT
            )
        except Exception as e:
            logger.warning(
                f"Error when generating with cache, trying without cache: '{e}'"
            )
            output = llm._generate(
                messages, request_timeout=env.WDOC_LLM_REQUEST_TIMEOUT
            )
        if output.generations[0].generation_info is None:
            assert "fake-list-chat-model" in llm._get_llm_string()
            finish = "stop"
        else:
            finish = output.generations[0].generation_info["finish_reason"]
            assert finish == "stop", f"Unexpected finish_reason: '{finish}'"
            assert len(output.generations) == 1
        out = output.generations[0].text
        if output.llm_output:  # only present if not caching
            new_p = output.llm_output["token_usage"]["prompt_tokens"]
            new_c = output.llm_output["token_usage"]["completion_tokens"]
            new_r = output.llm_output["token_usage"]["total_tokens"] - (new_p + new_c)
            logger.debug(
                "LLM token usage output for that completion: "
                + str(output.llm_output["token_usage"])
            )
        else:
            new_p = 0
            new_c = 0
            new_r = 0
        token_details["prompt"] += new_p
        token_details["completion"] += new_c
        token_details["internal_reasoning"] += new_r

        # the callback need to be updated manually when _generate is called
        llm.callbacks[0].prompt_tokens += new_p
        llm.callbacks[0].completion_tokens += new_c
        llm.callbacks[0].internal_reasoning_tokens += new_r
        llm.callbacks[0].total_tokens += new_p + new_c + new_r

        parsed = thinking_answer_parser(out)
        if verbose and parsed["thinking"]:
            logger.info("Thinking: " + parsed["thinking"])

        output_lines = parsed["answer"].rstrip().splitlines(keepends=True)

        # Remove first line if:
        # - it contains "a deep breath"
        # - it starts with "i'll summarize" (case insensitive)
        # - it's a bullet point containing these phrases
        if output_lines:
            first_line = output_lines[0].lower()
            should_remove = (
                ("deep breath" in first_line and len(first_line) < 20)
                or (first_line.startswith("i'll summarize") and len(first_line) < 20)
                or (
                    first_line.strip().startswith("- ")
                    and ("deep breath" in first_line or "i'll summarize" in first_line)
                    and (len(first_line) < 20)
                )
            )
            if should_remove:
                output_lines = output_lines[1:]

        for il, ll in enumerate(output_lines):
            # remove if contains no alphanumeric character
            if not any(char.isalpha() for char in ll.strip()):
                output_lines[il] = None
                continue

            ll = ll.rstrip()

            # replace tabs by 4 spaces
            ll = ll.replace("\t", "    ")
            ll = ll.replace("	", "    ")

            stripped = ll.lstrip()

            ll = ll.replace("- • ", "- ").replace("• ", "- ")  # occasional bad md

            # if a line starts with * instead of -, fix it
            if stripped.startswith("* "):
                ll = ll.replace("*", "-", 1)

            stripped = ll.lstrip()
            # beginning with long dash
            if stripped.startswith("—"):
                ll = ll.replace("—", "-")

            # begin by '-' but not by '- '
            stripped = ll.lstrip()
            if stripped.startswith("-") and not stripped.startswith("- "):
                ll = ll.replace("-", "- ", 1)

            # if a line does not start with - fix it
            stripped = ll.lstrip()
            if not stripped.startswith("- "):
                ll = ll.replace(stripped[0], "- " + stripped[0], 1)

            ll = ll.replace("****", "")

            # if contains uneven number of bold markers
            if ll.count("**") % 2 == 1:
                ll += "**"  # end the bold
            # and italic
            if ll.count("*") % 2 == 1:
                ll += "*"  # end the italic

            output_lines[il] = ll

        good_lines = [li for li in output_lines if (li and li.replace("-", "").strip())]
        output_text = "\n".join(good_lines)

        if verbose:
            logger.info(output_text)

        assert "{previous_summary}" in PREV_SUMMARY_TEMPLATE
        previous_summary = PREV_SUMMARY_TEMPLATE.replace(
            "{previous_summary}",
            "\n".join(good_lines[-10:]),
        )

        summaries.append(output_text)

    # combine summaries as one string separated by markdown separator
    n = len(summaries)
    if n > 1:
        outtext = f"- Chunk 1/{n}\n"
        for i, s in enumerate(summaries):
            outtext += s + "\n"
            if n > 1 and s != summaries[-1]:
                outtext += f"- ---\n- Chunk {i + 2}/{n}\n"
    else:
        outtext = "\n".join(summaries)

    return outtext.rstrip(), n, token_details
