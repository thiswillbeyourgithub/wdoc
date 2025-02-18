import sys
from pathlib import Path

import pytest
from langchain_core.documents.base import Document

from wdoc.utils.embeddings import load_embeddings_engine, test_embeddings
from wdoc.utils.misc import ModelName
from wdoc.wdoc import wdoc


@pytest.mark.basic
def test_wdoc_version():
    """Test that wdoc has a valid version string."""
    assert isinstance(wdoc.VERSION, str)
    assert len(wdoc.VERSION.split(".")) == 3


@pytest.mark.basic
def test_fail_parse_small_file_text(sample_text_file):
    """Test that a too small text file parsing fails."""
    # should fail because the file is too small
    with pytest.raises(Exception):
        wdoc.parse_file(
            path=str(sample_text_file), filetype="txt", debug=False, verbose=False
        )


@pytest.mark.basic
def test_parse_file_text(sample_text_file):
    """Test basic text file parsing."""
    # make a bigger text file
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        debug=False,
        verbose=False,
        format="langchain",
    )
    assert len(docs) > 0
    assert docs[0].page_content.startswith("This is a test document")
    assert "multiple lines" in docs[0].page_content


@pytest.mark.basic
def test_parse_file_formats(sample_text_file):
    """Test text-only output from parse_file."""
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
        debug=False,
        verbose=False,
    )
    assert isinstance(docs, list), type(docs)
    assert len(docs) == 1, len(docs)
    assert all(isinstance(d, Document) for d in docs), ",".join(type(d) for d in docs)
    doc = docs[0]
    assert isinstance(doc, Document), type(text)
    assert doc.page_content.startswith("This is a test document"), doc
    assert "multiple lines" in doc.page_content, doc.page_content

    ld = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain_dict",
        debug=False,
        verbose=False,
    )
    assert isinstance(ld, list), type(ld)
    for ldd in ld:
        assert isinstance(ldd, dict), ldd
        assert "page_content" in ldd, ldd
        assert "metadata" in ldd, ldd

    text = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="text",
        debug=False,
        verbose=False,
    )
    assert isinstance(text, str), type(text)

    xml = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        format="xml",
        debug=False,
        verbose=False,
    )
    assert isinstance(xml, str), type(xml)

    assert xml != text


@pytest.mark.basic
def test_invalid_filetype():
    """Test that invalid filetype raises an error."""
    with pytest.raises(Exception):
        wdoc.parse_file(
            path="dummy.txt", filetype="invalid_type", debug=False, verbose=False
        )


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban():
    """Test summarization of Tim Urban's procrastination video."""
    _ = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        modelname="openai/gpt-4o",
        query_eval_modelname="openai/gpt-4o-mini",
        embed_model="openai/text-embedding-3-small",
        # filetype="youtube",
        filetype="auto",
        debug=False,
        verbose=False,
        import_mode=True,
    )


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban():
    """Test query task on Tim Urban's procrastination video."""
    _ = wdoc(
        task="query",
        query="What university did the author go to?",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        modelname="openai/gpt-4o",
        query_eval_modelname="openai/gpt-4o-mini",
        embed_model="openai/text-embedding-3-small",
        # filetype="youtube",
        filetype="auto",
        debug=False,
        verbose=False,
        import_mode=True,
    )


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_openai_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName("openai/text-embedding-3-small"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    test_embeddings(emb)


@pytest.mark.basic
def test_ollama_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName("ollama/bge-m3"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    test_embeddings(emb)


# @pytest.mark.basic
# def test_hf_embeddings():
#     emb = load_embeddings_engine(
#         modelname=ModelName("huggingface/sentence-transformers/paraphrase-MiniLM-L6-v2"),
#         cli_kwargs={},
#         api_base=None,
#         embed_kwargs={},
#         private=False,
#         do_test=True,
#     )
#     test_embeddings(emb)
