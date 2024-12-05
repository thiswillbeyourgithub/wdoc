import pytest
from pathlib import Path
from wdoc.wdoc import wdoc

def test_wdoc_version():
    """Test that wdoc has a valid version string."""
    assert isinstance(wdoc.VERSION, str)
    assert len(wdoc.VERSION.split(".")) == 3

def test_fail_parse_small_file_text(sample_text_file):
    """Test that a too small text file parsing fails."""
    # should fail because the file is too small
    with pytest.raises(Exception):
        wdoc.parse_file(
            path=str(sample_text_file),
            filetype="txt",
            debug=False,
            verbose=False
        )

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
        verbose=False
    )
    assert len(docs) > 0
    assert docs[0].page_content.startswith("This is a test document")
    assert "multiple lines" in docs[0].page_content

def test_parse_file_only_text(sample_text_file):
    """Test text-only output from parse_file."""
    f = Path(sample_text_file)
    content = f.read_text()
    f.write_text(50 * (content + "\n"))
    text = wdoc.parse_file(
        path=str(sample_text_file),
        filetype="txt",
        only_text=True,
        debug=False,
        verbose=False
    )
    assert isinstance(text, str)
    assert text.startswith("This is a test document")
    assert "multiple lines" in text

def test_invalid_filetype():
    """Test that invalid filetype raises an error."""
    with pytest.raises(Exception):
        wdoc.parse_file(
            path="dummy.txt",
            filetype="invalid_type",
            debug=False,
            verbose=False
        )
