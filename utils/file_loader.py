import random
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
import tiktoken

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS

from .misc import loaddoc_cache, html_to_text, hasher
from .logger import whi, yel, red, log
from utils.misc import embed_cache

clozeregex = re.compile(r"{{c\d+::|}}")
tokenize = tiktoken.encoding_for_model("gpt-3.5-turbo").encode


def len_split(tosplit):
    return len(tokenize(tosplit))


text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
        chunk_size=2000,  # default 4000
        chunk_overlap=350,  # default 200
        length_function=len_split,
        )


def cloze_stripper(clozed):
    clozed = re.sub(clozeregex, " ", clozed)
    return clozed


def load_doc(filetype, debug, **kwargs):
    """load the input"""

    if filetype in ["path_list", "recursive"]:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]

        if filetype == "recursive":
            assert "pattern" in kwargs, "missing 'pattern' key in args"
            assert "recursed_filetype" in kwargs, "missing 'recursed_filetype' in args"
            assert kwargs["recursed_filetype"] not in [
                    "recursive", "path_list", "youtube", "anki",
                    ], "'recursed_filetype' cannot be 'recursive', 'path_list', 'anki' or 'youtube'"
            pattern = kwargs["pattern"]

            doclist = [p for p in Path(path).rglob(pattern)]
            doclist = [str(p).strip() for p in doclist if p.is_file()]
            doclist = [p for p in doclist if p]

            # randomize order to even out the progress bar
            doclist = sorted(doclist, key=lambda x: random.random())

            def threaded_load_item(filetype, item, kwargs):
                meta = kwargs.copy()
                meta["path"] = item
                meta["filetype"] = meta["recursed_filetype"]
                assert Path(meta["path"]).exists(), f"file '{item}' does not exist"
                del meta["pattern"]
                try:
                    return load_doc(
                            debug=debug,
                            **meta,
                            )
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    return None

        elif filetype == "path_list":
            doclist = str(Path(path).read_text()).splitlines()
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#")]

            def threaded_load_item(filetype, item, kwargs):
                meta = json.loads(item.strip())
                assert isinstance(meta, dict), f"meta from line '{item}' is not dict but '{type(meta)}'"
                assert "filetype" in meta, "no key 'filetype' in meta"
                try:
                    return load_doc(
                            debug=debug,
                            **meta,
                            )
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    return None

        else:
            raise ValueError(filetype)

        if "include" in kwargs:
            for inc in kwargs["include"]:
                if inc == inc.lower():
                    inc = re.compile(inc, flags=re.IGNORECASE)
                else:
                    inc = re.compile(inc)
                doclist = [p for p in doclist if not re.search(inc, p)]
            del kwargs["include"]

        if "exclude" in kwargs:
            for exc in kwargs["exclude"]:
                if exc == exc.lower():
                    exc = re.compile(exc, flags=re.IGNORECASE)
                else:
                    exc = re.compile(exc)
                doclist = [p for p in doclist if not re.search(exc, p)]
            del kwargs["exclude"]

        assert doclist, "empty list of documents to load!"

        # use multithreading only if recursive
        results = Parallel(
                n_jobs=3 if len(doclist) >= 3 else 1,
                backend="threading" if not debug and filetype == "recursive" else "sequential",
                )(delayed(threaded_load_item)(filetype, doc, kwargs
                    ) for doc in tqdm(doclist, desc="loading list of documents"))

        results = [r for r in results if r]
        assert results, "Empty results after loading documents"
        n = len(doclist) - len(results)
        if n:
            red(f"There were errors when loading documents: '{n}' documents failed")
        docs = []
        [docs.extend(x) for x in results if x]
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
        docs = loader.load()
        docs = text_splitter.transform_documents(docs)

    elif filetype == "pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        loader = PyPDFLoader(path)
        docs = loader.load()
        docs = loaddoc_cache.eval(text_splitter.transform_documents, docs)

    elif filetype == "anki":
        for nk in ["anki_deck", "anki_notetype", "anki_profile", "anki_fields"]:
            assert nk in kwargs, f"Missing '{nk}' in arguments from load_doc"
        profile = kwargs["anki_profile"]
        deck = kwargs["anki_deck"]
        notetype = kwargs["anki_notetype"]
        fields = kwargs["anki_fields"]
        whi(f"Loading anki profile: '{profile}'")
        original_db = akp.find_db(user=profile)
        name = f"{profile}".replace(" ", "_")
        temp_db = shutil.copy(original_db, f"./.cache/anki_collection_{name.replace('/', '_')}")
        col = akp.Collection(path=temp_db)
        cards = col.cards.merge_notes()
        cards.loc[cards['codeck']=="", 'codeck'] = cards['cdeck'][cards['codeck']==""]
        cards["codeck"] = cards["codeck"].apply(lambda x: x.replace("\x1f", "::"))
        cards = cards[cards["codeck"].str.startswith(deck)]
        cards["nmodel"] = cards["nmodel"].apply(lambda x: x.lower())
        cards = cards[cards["nmodel"].str.startswith(notetype)]

        cards["mid"] = col.cards.mid.loc[cards.index]
        mid2fields = akp.raw.get_mid2fields(col.db)
        mod2mid = akp.raw.get_model2mid(col.db)
        cards["fields_name"] = cards["mid"].apply(lambda x: mid2fields[x])
        assert cards.index.tolist(), "empty dataframe!"
        cards["fields_dict"] = cards.apply(
                lambda x: {
                    k: html_to_text(cloze_stripper(v), issoup=False).strip()
                    for k, v in zip(x["fields_name"], x["nflds"])
                    if k.lower() in fields
                    },
                axis=1)
        cards["text"] = cards["fields_dict"].apply(
            lambda x: "\n".join(
                f"{k}: {x[k]}" for k in fields
                if x[k]
                ))

        cards = cards.sort_index()

        # load each card as a single document
        # docs = [Document(page_content=t) for t in cards["text"].tolist()]

        # turn all cards into a single wall of text then use text_splitter
        # full_df = "\n\n\n\n".join(cards["text"].tolist())
        # texts = loaddoc_cache.eval(text_splitter.split_text, full_df)
        # docs = [Document(page_content=t) for t in texts]

        # turn each X cards into one document
        window_size = 5
        index_list = cards.index.tolist()
        n = len(index_list)
        cards["text_concat"] = ""
        for i in tqdm(range(len(index_list)), desc="combining anki cards"):
            for w in range(window_size):
                if i + window_size < n:
                    cards.loc[index_list[i], "text_concat"] += cards.loc[index_list[i+w], "text"]
                else:
                    cards.loc[index_list[i], "text_concat"] += cards.loc[index_list[i+w-window_size], "text"]
        docs = [Document(page_content=t) for t in cards["text_concat"]]

        for i in range(len(docs)):
            docs[i].metadata["anki_profile"] = profile
            docs[i].metadata["anki_deck"] = deck
            docs[i].metadata["anki_notetype"] = notetype
            docs[i].metadata["path"] = f"Anki profile '{profile}' deck '{deck}'"

    elif filetype == "string":
        whi("Loading string")
        content = prompt(
                "Paste your text content here then press esc+enter or meta+enter:\n>",
                multiline=True,
                )
        log.info(f"Pasted string input:\n{content}")
        texts = loaddoc_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]

    elif filetype == "txt":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading txt: '{path}'")
        whi(path)
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        texts = loaddoc_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]

    # add metadata
    for i in range(len(docs)):
        docs[i].metadata["hash"] = hasher(docs[i].page_content)
        docs[i].metadata["head"] = str(docs[i].page_content)[:50]
        docs[i].metadata["filetype"] = filetype
        if "path" not in docs[i].metadata and "path" in locals():
            docs[i].metadata["path"] = path

        # if html, parse it
        soup = BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = html_to_text(soup, issoup=True)

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

    assert docs, "empty list of loaded documents!"
    docs = [d for d in docs if d.page_content]
    assert docs, "empty list of loaded documents after removing empty docs!"

    return docs


def load_embeddings(sbert_model, loadfrom, saveas, debug, loaded_docs):
    """loads embeddings for each document"""
    embeddings = SentenceTransformerEmbeddings(
            model_name=sbert_model,
            encode_kwargs={
                "batch_size": 1,
                "show_progress_bar": False,
                },
            )

    # reload passed embeddings
    if loadfrom:
        red("Reloading documents and embeddings from file")
        path = Path(loadfrom)
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(str(path), embeddings)
        return db

    red("\nLoading embeddings.")

    docs = loaded_docs
    if len(docs) >= 50:
        docs = sorted(docs, key=lambda x: random.random())
    (embed_cache / sbert_model).mkdir(exist_ok=True)

    def get_embedding(doc, embeddings, embed_cache):
        hashcheck = doc.metadata["hash"]
        if (embed_cache / sbert_model / hashcheck).exists():
            try:
                temp = FAISS.load_local(str(embed_cache / sbert_model / hashcheck), embeddings)
                whi(f"Loaded from cache '{doc.metadata['path']}'")
                return temp, hashcheck, doc.metadata['path']
            except Exception as err:
                red(f"Error (will compute embedding instead of loading form file): '{err}'")

        whi("Computing embeddings")
        temp = FAISS.from_documents([doc], embeddings)
        temp.save_local(str(embed_cache / sbert_model / hashcheck))
        return temp, hashcheck, doc.metadata['path']

    results = Parallel(
            n_jobs=3,
            backend=threading" if not debug else "sequential",
            )(delayed(get_embedding)(doc, embeddings, embed_cache) for doc in tqdm(docs, desc="embedding documents"))

    # merge the results
    done_list = set()
    db = None
    for temp, hashcheck, path in results:
        (embed_cache / sbert_model / hashcheck).touch()  # this way we know what files where not used in a long time
        if db is None:
            db = temp
        else:
            if hashcheck not in done_list:
                try:
                    db.merge_from(temp)
                    done_list.add(hashcheck)
                except Exception as err:
                    red(f"Error when merging index: '{err}'")
            else:
                whi(f"File with path '{path}' with hash '{hashcheck}' was already added, skipping.")

    # saving embeddings
    path = Path(saveas)
    db.save_local(str(path))

    return db
