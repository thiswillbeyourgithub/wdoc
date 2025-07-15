import os
import subprocess
import tempfile
import hashlib
from pathlib import Path

import pytest
import requests
from langchain_core.documents.base import Document

# Environment setup for parsing tests
os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"
os.environ["WDOC_TYPECHECKING"] = "crash"

from wdoc.wdoc import wdoc


@pytest.mark.basic
def test_fail_parse_small_file_text(sample_text_file):
    """Test that a too small text file parsing fails."""
    # should fail because the file is too small
    with pytest.raises(Exception):
        wdoc.parse_doc(
            path=str(sample_text_file),
            filetype="txt",
        )


@pytest.mark.basic
def test_parse_doc_text(sample_text_file):
    """Test basic text file parsing."""
    # make a bigger text file
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
    )
    assert len(docs) > 0
    assert docs[0].page_content.startswith("This is a test document")
    assert "multiple lines" in docs[0].page_content


@pytest.mark.basic
def test_parse_doc_formats(sample_text_file):
    """Test text-only output from parse_doc."""
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    docs = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain",
    )
    assert isinstance(docs, list), type(docs)
    assert len(docs) == 1, len(docs)
    assert all(isinstance(d, Document) for d in docs), ",".join(type(d) for d in docs)
    doc = docs[0]
    assert isinstance(doc, Document), type(doc)
    assert doc.page_content.startswith("This is a test document"), doc
    assert "multiple lines" in doc.page_content, doc.page_content

    ld = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="langchain_dict",
    )
    assert isinstance(ld, list), type(ld)
    for ldd in ld:
        assert isinstance(ldd, dict), ldd
        assert "page_content" in ldd, ldd
        assert "metadata" in ldd, ldd

    text = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="text",
    )
    assert isinstance(text, str), type(text)

    xml = wdoc.parse_doc(
        path=str(sample_text_file),
        filetype="txt",
        format="xml",
    )
    assert isinstance(xml, str), type(xml)

    assert xml != text


@pytest.mark.basic
def test_invalid_filetype():
    """Test that invalid filetype raises an error."""
    with pytest.raises(Exception):
        wdoc.parse_doc(
            path="dummy.txt",
            filetype="invalid_type",
        )


@pytest.mark.basic
def test_parse_online_pdf():
    """Test parsing an online PDF about situational awareness."""
    docs = wdoc.parse_doc(
        path="https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf",
        filetype="online_pdf",
        format="langchain",
    )
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    assert any("alphago" in d.page_content.lower() for d in docs)


@pytest.mark.basic
def test_parse_docx():
    """Test parsing a DOCX file."""
    # Create temporary file and download sample DOCX file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    url = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_500KB_DOCX.docx"

    # Download file using requests with proper headers to avoid 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Write content to temporary file
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    # Verify SHA512 checksum
    expected_hash = "64b73b409688cc5b5675c07d9df4b83d353fa85026a9d686d6725e50f388930e1d57c56cc6cfebd5f2cecc06d7ef89ae7495bd5411ca0eac4b0df63a7d6c82dc"
    with open(tmp_path, "rb") as f:
        file_hash = hashlib.sha512(f.read()).hexdigest()
    assert (
        file_hash == expected_hash
    ), f"File hash {file_hash} does not match expected {expected_hash}"

    try:
        # Parse the file
        docs = wdoc.parse_doc(
            path=tmp_path,
            # filetype="word",
            format="langchain",
        )

        # Verify output
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # Check that we got some actual content
        assert any(len(d.page_content.strip()) > 0 for d in docs)

    finally:
        # Cleanup
        os.unlink(tmp_path)


@pytest.mark.basic
def test_parse_nytimes():
    """Test parsing the NYTimes homepage."""
    docs = wdoc.parse_doc(
        path="https://www.nytimes.com/",
        filetype="auto",
        format="langchain",
    )
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    # Check that we got some actual content
    assert any(len(d.page_content.strip()) > 0 for d in docs)
