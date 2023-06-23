import openai
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
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS

from .misc import loaddoc_cache, html_to_text, hasher
from .logger import whi, yel, red, log
from utils.misc import embed_cache

charac_regex = re.compile(r"[^\w\s]")
clozeregex = re.compile(r"{{c\d+::|}}")
tokenize = tiktoken.encoding_for_model("gpt-3.5-turbo").encode


def len_split(tosplit):
    return len(tokenize(tosplit))


text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
        chunk_size=1024,  # default 4000
        chunk_overlap=386,  # default 200
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
            for i, d in enumerate(doclist):
                keep = True
                for inc in kwargs["include"]:
                    if not re.search(inc, d):
                        keep = False
                if not keep:
                    doclist[i] = None
            doclist = [d for d in doclist if d]
            del kwargs["include"]

        if "exclude" in kwargs:
            for exc in kwargs["exclude"]:
                doclist = [d for d in doclist if not re.search(exc, d)]
            del kwargs["exclude"]

        assert doclist, "empty list of documents to load!"

        # use multithreading only if recursive
        results = Parallel(
                n_jobs=4 if len(doclist) >= 3 else 1,
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

        #loader = PyPDFLoader(path)
        loader = PDFMinerLoader(path)
        content  = loader.load()
        content = "\n".join([d.page_content for d in content])
        texts = loaddoc_cache.eval(text_splitter.split_text, content)
        docs = [Document(page_content=t) for t in texts]

        # source: https://python.langchain.com/docs/modules/data_connection/document_loaders/how_to/pdf
        # loader = PDFMinerPDFasHTMLLoader(path)
        # data = loader.load()[0]   # entire pdf is loaded as a single Document
        # def parsePDFasHTML(data):
        #     soup = BeautifulSoup(data.page_content,'html.parser')
        #     content = soup.find_all('div')
        #     cur_fs = None
        #     cur_text = ''
        #     snippets = []   # first collect all snippets that have the same font size
        #     for c in content:
        #         sp = c.find('span')
        #         if not sp:
        #             continue
        #         st = sp.get('style')
        #         if not st:
        #             continue
        #         fs = re.findall('font-size:(\d+)px',st)
        #         if not fs:
        #             continue
        #         fs = int(fs[0])
        #         if not cur_fs:
        #             cur_fs = fs
        #         if fs == cur_fs:
        #             cur_text += c.text
        #         else:
        #             snippets.append((cur_text,cur_fs))
        #             cur_fs = fs
        #             cur_text = c.text
        #     snippets.append((cur_text,cur_fs))
        #     # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
        #     # headers/footers in a PDF appear on multiple pages so if we find duplicatess safe to assume that it is redundant info)
        #     cur_idx = -1
        #     semantic_snippets = []
        #     # Assumption: headings have higher font size than their respective content
        #     for s in snippets:
        #         # if current snippet's font size > previous section's heading => it is a new heading
        #         if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
        #             metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
        #             metadata.update(data.metadata)
        #             semantic_snippets.append(Document(page_content='',metadata=metadata))
        #             cur_idx += 1
        #             continue
        #         
        #         # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        #         # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        #         if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
        #             semantic_snippets[cur_idx].page_content += s[0]
        #             semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
        #             continue
        #         
        #         # if current snippet's font size > previous section's content but less tha previous section's heading than also make a new 
        #         # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
        #         metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
        #         metadata.update(data.metadata)
        #         semantic_snippets.append(Document(page_content='', metadata=metadata))
        #         cur_idx += 1
        #     docs = text_splitter.transform_documents(semantic_snippets)
        #     return docs
        # docs = loaddoc_cache.eval(parsePDFasHTML, data)


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
        cards = cards[~cards["text"].str.contains("[IMAGE]")]

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
                    cards.loc[index_list[i], "text_concat"] += "\n\n" + cards.loc[index_list[i+w], "text"]
                else:
                    cards.loc[index_list[i], "text_concat"] += "\n\n" + cards.loc[index_list[i+w-window_size], "text"]
        docs = [Document(page_content=t) for t in cards["text_concat"]]
        assert docs, "List of loaded anki document is empty!"

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
        path = "user_string"

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


def load_embeddings(embed_model, loadfrom, saveas, debug, loaded_docs, kwargs):
    """loads embeddings for each document"""
    embed_args = {}

    if embed_model == "openai":
        red("Using openai embedding model")
        assert Path("API_KEY.txt").exists(), "No API_KEY.txt found"

        embeddings = OpenAIEmbeddings(
                openai_api_key = str(Path("API_KEY.txt").read_text()).strip()
                )

    else:
        embeddings = SentenceTransformerEmbeddings(
                model_name=embed_model,
                encode_kwargs={
                    "batch_size": 1,
                    "show_progress_bar": False,
                    "normalize_embeddings": True,
                    },
                )
        if "stopwords" in kwargs:
            embed_args["stopwords"] = kwargs["stopwords"]

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
    (embed_cache / embed_model).mkdir(exist_ok=True)

    def get_embedding(doc, embeddings, embed_cache, embed_args=embed_args):
        hashcheck = doc.metadata["hash"]
        if (embed_cache / embed_model / hashcheck).exists():
            try:
                temp = FAISS.load_local(str(embed_cache / embed_model / hashcheck), embeddings)
                whi(f"Loaded from cache '{doc.metadata['path']}'")
                return temp, hashcheck, doc.metadata['path']
            except Exception as err:
                red(f"Error (will compute embedding instead of loading form file): '{err}'")

        whi("Computing embeddings")
        if "stopwords" in embed_args:
            prev = doc.page_content
            doc.page_content = re.sub(charac_regex, " ", doc.page_content.lower())
            for reg in embed_args["stopwords"]:
                doc.page_content = re.sub(reg, " ", doc.page_content)
        temp = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        if "stopwords" in embed_args:
            for k in temp.docstore.__dict__.keys():
                for kk in temp.docstore.__dict__[k].keys():
                    temp.docstore.__dict__[k][kk].page_content = prev
        temp.save_local(str(embed_cache / embed_model / hashcheck))
        return temp, hashcheck, doc.metadata['path']

    results = Parallel(
            n_jobs=3,
            backend="threading" if not debug else "sequential",
            )(delayed(get_embedding)(doc, embeddings, embed_cache) for doc in tqdm(docs, desc="embedding documents"))

    # merge the results
    done_list = set()
    db = None
    for temp, hashcheck, path in results:
        (embed_cache / embed_model / hashcheck).touch()  # this way we know what files where not used in a long time
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
