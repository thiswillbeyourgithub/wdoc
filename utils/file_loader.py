import shutil
import ankipandas as akp
import ftfy
from bs4 import BeautifulSoup
from pathlib import Path
import re
from tqdm import tqdm
import json
from prompt_toolkit import prompt
from joblib import Parallel, delayed

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import FAISS

from .misc import split_cache, html_to_text, hasher
from .logger import whi, yel, red, log
from utils.misc import docstore_cache

text_splitter = CharacterTextSplitter()


def load_documents(**kwargs):
    red("\nLoading documents.")
    if "loadfrom" not in kwargs:
        kwargs["loaded_docs"] = _load_doc(**kwargs)
        whi(f"\n\nLoaded '{len(kwargs['loaded_docs'])}' documents")
    if kwargs["task"] == "query":
        kwargs["loaded_embeddings"] = _load_embeddings(**kwargs)
    return kwargs


def _load_doc(**kwargs):
    """load the input"""
    filetype = kwargs["filetype"]

    if filetype in ["path_list", "recursive"]:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(filetype)
        if filetype == "recursive":
            assert "pattern" in kwargs, "missing 'pattern' key in args"
            assert "recursed_filetype" in kwargs, "missing 'recursed_filetype' in args"
            assert kwargs["recursed_filetype"] not in [
                    "recursive", "path_list", "youtube"
                    ], "'recursed_filetype' cannot be 'recursive', 'path_list' or 'youtube'"
            pattern = kwargs["pattern"]
            doclist = [p for p in Path(path).rglob(pattern)]
            doclist = [str(p) for p in doclist if p.is_file()]
            if "exclude" in kwargs:
                for exc in kwargs["exclude"]:
                    if exc == exc.lower():
                        exc = re.compile(exc, flags=re.IGNORECASE)
                    else:
                        exc = re.compile(exc)
                    doclist = [p for p in doclist if not re.search(exc, p)]
                del kwargs["exclude"]
            assert doclist, "empty recursive search!"

        elif filetype == "path_list":
            doclist = str(Path(path).read_text()).splitlines()
        else:
            raise ValueError(filetype)

        docs = []
        for item in tqdm(doclist, desc="loading list of documents"):
            item = item.strip()
            if not item:
                continue
            if item.startswith("#"):
                continue
            if filetype == "path_list":
                meta = json.loads(item.strip())
                assert isinstance(meta, dict), f"meta from line '{item}' is not dict but '{type(meta)}'"
                assert "filetype" in meta, "no key 'filetype' in meta"
            elif filetype == "recursive":
                meta = kwargs.copy()
                meta["filetype"] = kwargs["recursed_filetype"]
                meta["path"] = item
                assert Path(meta["path"]).exists(), f"file '{item}' does not exist"
                del meta["pattern"]
            else:
                raise ValueError(filetype)
            docs.extend(_load_doc(**meta))
        return docs

    if filetype == "youtube":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading youtube: '{path}'")
        loader = YoutubeLoader.from_youtube_url(
                path,
                add_video_info=True,
                language=[kwargs["language"]],
                translation=kwargs["translation"],
                )
        loader.load()
        docs = loader.load_and_split()

    elif filetype == "pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        loader = PyPDFLoader(path)
        # try:
        docs = split_cache.eval(loader.load_and_split)
        # except Exception as err:
        #     whi(f"Error when using cache to load '{path}': '{err}'")
        #     docs = loader.load_and_split()

    elif filetype == "anki":
        needed_keys = ["anki_deck", "anki_notetype", "anki_profile"]
        for nk in needed_keys:
            assert nk in kwargs, f"Missing '{nk}' in arguments from load_doc"
        profile = kwargs["anki_profile"]
        deck = kwargs["anki_deck"]
        whi(f"Loading anki profile: '{profile}'")
        original_db = akp.find_db(user=profile)
        name = f"{profile}".replace(" ", "_")
        temp_db = shutil.copy(original_db, f"./.cache/anki_collection_{name.replace('/', '_')}")
        col = akp.Collection(path=temp_db)
        cards = col.cards.merge_notes()
        cards["codeck"] = cards["codeck"].apply(lambda x: x.replace("\x1f", "::"))
        cards = cards[cards["codeck"].str.startswith(deck)]
        cards = cards[cards["nmodel"].str.startswith(kwargs["anki_notetype"])]
        cards["fields"] = cards["nflds"].apply(lambda x: "\n\n".join(x)[:500])
        cards["fields"] = cards["fields"].apply(lambda x: html_to_text(x, issoup=False))
        loader = DataFrameLoader(
            cards,
            page_content_column="fields",
            )
        docs = loader.load()

        for i in range(len(docs)):
            docs[i].metadata["anki_profile"] = profile
            docs[i].metadata["anki_deck"] = deck
            docs[i].metadata["anki_notetype"] = kwargs["anki_notetype"]
            docs[i].metadata["path"] = f"Anki profile '{profile}' deck '{deck}'"

    elif filetype == "string":
        whi("Loading string")
        content = prompt(
                "Paste your text content here then press esc+enter or meta+enter:\n>",
                multiline=True,
                )
        log.info(f"Pasted string input:\n{content}")
        texts = split_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]

    elif filetype == "txt":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading txt: '{path}'")
        whi(path)
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        texts = split_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]

    # add metadata
    for i in range(len(docs)):
        docs[i].metadata["hash"] = hasher(docs[i].page_content)
        docs[i].metadata["head"] = str(docs[i].page_content)[:100]
        docs[i].metadata["filetype"] = filetype
        if "path" not in docs[i].metadata and "path" in locals():
            docs[i].metadata["path"] = path

        # if html, parse it
        soup = BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = html_to_text(soup, issoup=True)

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

    return docs


def _load_embeddings(**kwargs):
    """loads embeddings for each document"""
    embeddings = SentenceTransformerEmbeddings(
            model_name=kwargs["sbert_model"],
            encode_kwargs={
                "batch_size": 1,
                "show_progress_bar": False,
                },
            )

    # reload passed embeddings
    if "loadfrom" in kwargs:
        red("Reloading documents and embeddings from file")
        path = Path(kwargs["loadfrom"])
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(str(path), embeddings)
        return db

    red("\nLoading embeddings.")

    model_hash = hasher(kwargs["sbert_model"])
    docs = kwargs["loaded_docs"]

    def get_embedding(doc, embeddings, docstore_cache):
        hashcheck = f'{doc.metadata["hash"]}_{model_hash}'
        if (docstore_cache / hashcheck).exists():
            try:
                temp = FAISS.load_local(str(docstore_cache / hashcheck), embeddings)
                (docstore_cache / hashcheck).touch()  # this way we know what files where not used in a long time
                whi("Loaded from cache")
                return temp, hashcheck, doc.metadata['path']
            except Exception as err:
                red(f"Error (will compute embedding instead of loading form file): '{err}'")

        whi("Computing embeddings")
        temp = FAISS.from_documents([doc], embeddings)
        temp.save_local(str(docstore_cache / hashcheck))
        return temp, hashcheck, doc.metadata['path']

    results = Parallel(
            n_jobs=1,
            backend="threading",
            )(delayed(get_embedding)(doc, embeddings, docstore_cache) for doc in tqdm(docs, desc="embedding documents"))

    # merge the results
    done_list = set()
    db = None
    for temp, hashcheck, path in results:
        if db is None:
            db = temp
        else:
            if hashcheck not in done_list:
                db.merge_from(temp)
                done_list.add(hashcheck)
            else:
                whi(f"File '{path}' was already added, skipping.")


    # saving embeddings
    path = Path(kwargs["saveas"])
    db.save_local(str(path))

    return db
