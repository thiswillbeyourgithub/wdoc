import os
import tempfile
from pathlib import Path

import pytest

# apply the patch to make ollama work
os.environ["WDOC_APPLY_ASYNCIO_PATCH"] = "true"

import shutil
from pathlib import Path

# We have to reset the user cache dir when this file is executed to be able to test the caching
if (Path.cwd() / "wdoc_user_cache_dir").exists():
    shutil.rmtree((Path.cwd() / "wdoc_user_cache_dir").absolute())


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "basic: mark test as a basic test that doesn't require external services",
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring external API access"
    )


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    with open(file_path, "w") as f:
        f.write(
            "This is a test document.\nIt has multiple lines.\nFor testing purposes."
        )
    return file_path


@pytest.fixture(scope="session")
def sample_pdf_file(temp_dir):
    """Create a sample PDF file path for testing."""
    return temp_dir / "sample.pdf"
