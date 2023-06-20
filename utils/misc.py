import re
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
from joblib import Memory

Path(".cache").mkdir(exist_ok=True)
Path(".cache/embed_cache").mkdir(exist_ok=True)
Path(".cache/loaddoc_cache").mkdir(exist_ok=True)

embed_cache = Path(".cache/embed_cache/")
loaddoc_cache = Memory(".cache/loaddoc_cache/")

# remove cache files older than 90 days
#loaddoc_cache.reduce_size(age_limit=90)


def hasher(text):
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def html_to_text(html, issoup):
    """used to strip any html present in the text files"""
    if not issoup:
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        if "<img" in text:
            text = re.sub("<img src=.*?>", "[IMAGE]", text, flags=re.M|re.DOTALL)
        return text
    else:
        return html.get_text()
