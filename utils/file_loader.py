import time
import tempfile
import requests
import youtube_dl
from youtube_dl.utils import DownloadError, ExtractorError
import random
import shutil
import ankipandas as akp
import ftfy
from bs4 import BeautifulSoup
from goose3 import Goose
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
# from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from .misc import loaddoc_cache, html_to_text, hasher, embed_cache
from .logger import whi, yel, red, log
from .llm import RollingWindowEmbeddings

# rules used to attribute input to proper filetype. For example
# any link containing youtube will be treated as a youtube link
inference_rules = {
        # format:
        # key is output filtype, value is list of regex that if match
        # will return the key
        # the order of the keys is important
        "youtube_playlist": ["youtube.*playlist"],
        "youtube": ["youtube", "invidi"],
        "txt": [".txt$", ".md$"],
        "online_pdf": ["^http.*pdf.*"],
        "pdf": [".*pdf$"],
        "url": ["^http"],
        }

# compile the inference rules as regex
for k, v in inference_rules.items():
    try:
        for i, vv in enumerate(v):
            inference_rules[k][i] = re.compile(vv)
    except Exception as err:
        red(f"Exception when compiling inference_rules: '{err}' Disabling '{k}' inferences.")
        inference_rules[k] = []

# for reading length estimation
wpm = 200
average_word_length = 6

charac_regex = re.compile(r"[^\w\s]")  # for removing stopwords
clozeregex = re.compile(r"{{c\d+::|}}")  # for removing clozes in anki
markdownlink_regex = re.compile(r'\[.*?\]\((.*?)\)')  # to parse markdown links"
yt_link_regex = re.compile("youtube.*watch")  # to check that a youtube link is valid
tokenize = tiktoken.encoding_for_model("gpt-3.5-turbo").encode  # used to get token length estimation

def get_tkn_length(tosplit):
    return len(tokenize(tosplit))


def get_splitter(task):
    "we don't use the same text splitter depending on the task"
    if task == "query":
        text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
                chunk_size=3000,  # default 4000
                chunk_overlap=386,  # default 200
                length_function=get_tkn_length,
                )
    elif "summar" in task:
        text_splitter = RecursiveCharacterTextSplitter(
                separators=[".\n", ". ", " ", ""],
                chunk_size=700,
                chunk_overlap=150,
                length_function=get_tkn_length,
                )
    else:
        raise Exception(task)
    return text_splitter


def cloze_stripper(clozed):
    clozed = re.sub(clozeregex, " ", clozed)
    return clozed

min_token = 200
max_token = 50_000

def check_docs_tkn_length(docs, name):
    "checks that the number of tokens in the document is high enough, otherwise it probably means something went wrong."
    size = sum([get_tkn_length(d.page_content) for d in docs])
    if size <= min_token:
        raise Exception(f"The number of token from '{name}' is {size} <= {min_token} tokens, probably something went wrong?")
    elif size >= max_token:
        raise Exception(f"The number of token from '{name}' is {size} >= {max_token} tokens, probably something went wrong?")

def get_url_title(url):
    """if the title of the url is not loaded from the loader, trying as last
    resort with this one"""
    loader = WebBaseLoader(url, raise_for_status=True)
    docs = loader.load()
    if "title" in docs[0].metadata and docs[0].metadata["title"]:
        return docs[0].metadata["title"]
    else:
        return None

def load_doc(filetype, debug, task, **kwargs):
    """load the input"""
    text_splitter = get_splitter(task)

    if filetype == "infer":
        assert "path" in kwargs, "if filetype is infer, path should be supplied"
        for k, v in inference_rules.items():
            for vv in inference_rules[k]:
                if re.search(vv, kwargs["path"]):
                    filetype = k
                    break
            if filetype != "infer":
                break

    if filetype in ["json_list", "recursive", "link_file", "youtube_playlist"]:
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]

        if filetype == "recursive":
            whi(f"Loading recursive filetype: '{path}'")
            assert "pattern" in kwargs, "missing 'pattern' key in args"
            assert "recursed_filetype" in kwargs, "missing 'recursed_filetype' in args"
            assert kwargs["recursed_filetype"] not in [
                    "recursive", "json_list", "youtube", "anki",
                    ], "'recursed_filetype' cannot be 'recursive', 'json_list', 'anki' or 'youtube'"
            pattern = kwargs["pattern"]

            doclist = [p for p in Path(path).rglob(pattern)]
            doclist = [str(p).strip() for p in doclist if p.is_file()]
            doclist = [p for p in doclist if p]
            doclist = [p[1:].strip() if p.startswith("-") else p for p in doclist]

            # randomize order to even out the progress bar
            doclist = sorted(doclist, key=lambda x: random.random())

            n_thread = 20

            def threaded_load_item(filetype, item, kwargs):
                meta = kwargs.copy()
                meta["path"] = item
                meta["filetype"] = meta["recursed_filetype"]
                assert Path(meta["path"]).exists(), f"file '{item}' does not exist"
                del meta["pattern"]
                try:
                    return load_doc(
                            task=task,
                            debug=debug,
                            **meta,
                            )
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    return None

        elif filetype == "json_list":
            whi(f"Loading json_list: '{path}'")
            doclist = str(Path(path).read_text()).splitlines()
            doclist = [p[1:].strip() if p.startswith("-") else p for p in doclist]
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#")]

            # don't multithread this because a json line can itself be multithreaded.
            n_thread = 1

            def threaded_load_item(filetype, item, kwargs):
                meta = json.loads(item.strip())
                assert isinstance(meta, dict), f"meta from line '{item}' is not dict but '{type(meta)}'"
                assert "filetype" in meta, "no key 'filetype' in meta"
                try:
                    return load_doc(
                            task=task,
                            debug=debug,
                            **meta,
                            )
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    return None

        elif filetype == "link_file":
            whi(f"Loading link_file: '{path}'")
            doclist = str(Path(path).read_text()).splitlines()
            doclist = [p[1:].strip() if p.startswith("-") else p for p in doclist]
            doclist = [p.strip() for p in doclist if p.strip() and not p.strip().startswith("#") and "http" in p]
            doclist = [re.findall(markdownlink_regex, d)[0] if re.search(markdownlink_regex, d) else d for d in doclist]
            if task == "summarize_link_file":
                # if summarize, start from bottom
                doclist.reverse()

            n_thread = 20

            def threaded_load_item(filetype, item, kwargs):
                meta = kwargs.copy()
                meta["path"] = item
                if "http" not in item:
                    red(f"item does not appear to be a link: '{item}'")
                    return None
                meta["filetype"] = "infer"
                meta["subitem_link"] = item
                try:
                    return load_doc(
                            task=task,
                            debug=debug,
                            **meta,
                            )
                except Exception as err:
                    red(f"Error when loading '{item}': '{err}'")
                    return None

        elif filetype == "youtube_playlist":
            assert "path" in kwargs, "missing 'path' key in args"
            path = kwargs["path"]
            whi(f"Loading youtube playlist: '{path}'")
            video = load_youtube_playlist(path)

            kwargs["playlist_title"] = video['title'].strip().replace("\n", "")
            assert "duration" not in video, f'"duration" found when loading youtube playlist. This might not be a playlist: {path}'
            doclist = [ent["webpage_url"] for ent in video["entries"]]
            doclist = [li for li in doclist if re.search(yt_link_regex, li)]

            n_thread = 20

            def threaded_load_item(filetype, item, kwargs):
                meta = kwargs.copy()
                meta["path"] = item
                assert "http" in item, f"item does not appear to be a link: '{item}'"
                meta["filetype"] = "youtube"
                meta["subitem_link"] = item
                try:
                    return load_doc(
                            task=task,
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

        assert doclist, f"empty list of documents to load from filetype '{filetype}'"

        # use multithreading only if recursive
        if debug:
            n_thread = 1
        results = Parallel(
                n_jobs=n_thread,
                backend="threading",
                )(delayed(threaded_load_item)(filetype, doc, kwargs
                    ) for doc in tqdm(doclist, desc=f"loading documents from filetype '{filetype}' with n_thread={n_thread}"))

        results = [r for r in results if r]
        assert results, "Empty results after loading documents"
        n = len(doclist) - len(results)
        if n:
            red(f"There were errors when loading documents: '{n}' documents failed")
        docs = []
        [docs.extend(x) for x in results if x]

        size = sum([get_tkn_length(d.page_content) for d in docs])
        if size <= min_token:
            raise Exception(f"The number of token from '{path}' is {size} <= {min_token} tokens, probably something went wrong?")

    elif filetype == "youtube":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        if "\\" in path:
            red(f"Removed backslash found in '{path}'")
            path = path.replace("\\", "")
        assert re.search(yt_link_regex, path), f"youtube link is not valid: '{path}'"
        if "language" not in kwargs:
            lang = ["fr-FR", "fr", "en", "en-US", "en-UK"]
        else:
            lang = kwargs["language"]
        if "translation" not in kwargs:
            transl = "en"
        else:
            transl = kwargs["translation"]

        whi(f"Loading youtube: '{path}'")
        fyu = YoutubeLoader.from_youtube_url
        docs = cached_yt_loader(
                loader=fyu,
                path=path,
                add_video_info=True,
                language=lang,
                translation=transl,
                )
        docs = text_splitter.transform_documents(docs)

    elif filetype == "online_pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading online pdf: '{path}'")

        response = requests.get(path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()

        meta = kwargs.copy()
        meta["filetype"] == "pdf"
        meta["path"] = temp_file.name
        try:
            return load_doc(
                    task=task,
                    debug=debug,
                    **meta,
                    )
        except Exception as err:
            red(f"Error when parsing online pdf from {path} downloaded to {temp_file.name}: '{err}'")
            raise

    elif filetype == "pdf":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading pdf: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"

        try:
            loader = PDFMinerLoader(path)
            content  = loader.load()
            content = "\n".join([d.page_content for d in content])
            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]
            check_docs_tkn_length(docs, path)
        except Exception as err:
            red(f"Error when parsing '{path}' with PDFMiner. Using PyPDF as fallback.")
            loader = PyPDFLoader(path)
            content  = loader.load()
            content = "\n".join([d.page_content for d in content])
            texts = text_splitter.split_text(content)
            docs = [Document(page_content=t) for t in texts]
            check_docs_tkn_length(docs, path)

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

        # set window_size to X turn each X cards into one document, overlapping
        window_size = 1
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
        docs = text_splitter.transform_documents(docs)
        assert docs, "List of loaded anki document is empty!"

        for i in range(len(docs)):
            docs[i].metadata["anki_profile"] = profile
            docs[i].metadata["anki_deck"] = deck
            docs[i].metadata["anki_notetype"] = notetype
            docs[i].metadata["path"] = f"Anki profile '{profile}' deck '{deck}'"

        try:
            check_docs_tkn_length(docs, f"{filetype}: {profile}")
        except Exception as err:
            red(f"Number of token in anki document is surprising. Not quitting because anki can cause this: '{err}'")

    elif filetype == "string":
        whi("Loading string")
        content = prompt(
                "Paste your text content here then press esc+enter or meta+enter:\n>",
                multiline=True,
                )
        log.info(f"Pasted string input:\n{content}")
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
        path = "user_string"

    elif filetype == "txt":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading txt: '{path}'")
        assert Path(path).exists(), f"file not found: '{path}'"
        with open(path) as f:
            content = f.read()
        texts = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in texts]
        check_docs_tkn_length(docs, path)

    elif filetype == "url":
        assert "path" in kwargs, "missing 'path' key in args"
        path = kwargs["path"]
        whi(f"Loading url: '{path}'")

        # even if loading fails the title might be found so trying to keep
        # the first working title across trials
        if "title" not in kwargs or kwargs["title"] == "Untitled":
            title = None
        else:
            title = kwargs["title"]

        # try with playwright
        try:
            loader = PlaywrightURLLoader(urls=[path], remove_selectors=["header", "footer"])
            docs = text_splitter.transform_documents(loader.load())
            if not title and "title" in docs[0].metadata:
                title = docs[0].metadata["title"]
            check_docs_tkn_length(docs, path)

        # try with selenium firefox
        except Exception as err:
            red(f"Exception when using playwright to parse text: '{err}'\nUsing selenium firefox as fallback")
            try:
                loader = SeleniumURLLoader(urls=[path], browser="firefox")
                docs = text_splitter.transform_documents(loader.load())
                if not title and "title" in docs[0].metadata and docs[0].metadata["title"] != "No title found.":
                    title = docs[0].metadata["title"]
                check_docs_tkn_length(docs, path)

            # try with selenium chrome
            except Exception as err:
                red(f"Exception when using selenium firefox to parse text: '{err}'\nUsing selenium chrome as fallback")
                try:
                    loader = SeleniumURLLoader(urls=[path], browser="chrome")
                    docs = text_splitter.transform_documents(loader.load())
                    if not title and "title" in docs[0].metadata and docs[0].metadata["title"] != "No title found.":
                        title = docs[0].metadata["title"]
                    check_docs_tkn_length(docs, path)

                # try with goose
                except Exception as err:
                    red(f"Exception when using selenium chrome to parse text: '{err}'\nUsing goose as fallback")
                    try:
                        g = Goose()
                        article = g.extract(url=path)
                        text = article.cleaned_text
                        texts = text_splitter.split_text(text)
                        docs = [Document(page_content=t) for t in texts]
                        if not title:
                            if "title" in docs[0].metadata and docs[0].metadata["title"]:
                                title = docs[0].metadata["title"]
                            elif article.title:
                                title = article.title
                        check_docs_tkn_length(docs, path)

                    # try with html
                    except Exception as err:
                        red(f"Exception when using goose to parse text: '{err}'\nUsing html as fallback")
                        loader = WebBaseLoader(path, raise_for_status=True)
                        docs = text_splitter.transform_documents(loader.load())
                        if not title and "title" in docs[0].metadata and docs[0].metadata["title"]:
                            title = docs[0].metadata["title"]
                        check_docs_tkn_length(docs, path)

        # last resort, try to get the title from the most basic loader
        if not title:
            title = get_url_title(path)

        # store the title as metadata if missing
        if title:
            for d in docs:
                if "title" not in d.metadata or not d.metadata["title"]:
                    d.metadata["title"] = title
                else:
                    if d.metadata["title"] != title:
                        d.metadata["title"] = f"{title} - {d.metadata['title']}"

    else:
        raise Exception(red(f"Unsupported filetype: '{filetype}'"))

    # add and format metadata
    total_reading_length = None
    for i in range(len(docs)):
        # if html, parse it
        soup = BeautifulSoup(docs[i].page_content, "html.parser")
        if bool(soup.find()):
            docs[i].page_content = html_to_text(soup, issoup=True)

        # fix text just in case
        docs[i].page_content = ftfy.fix_text(docs[i].page_content)

        docs[i].metadata["hash"] = hasher(docs[i].page_content)

        if "Author" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["Author"]
            del docs[i].metadata["Author"]
        if "authors" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["authors"]
            del docs[i].metadata["authors"]
        if "Authors" in docs[i].metadata:
            docs[i].metadata["author"] = docs[i].metadata["Authors"]
            del docs[i].metadata["Authors"]
        if "filetype" not in docs[i].metadata:
            docs[i].metadata["filetype"] = filetype
        if "path" not in docs[i].metadata and "path" in locals():
            docs[i].metadata["path"] = path
        if "subitem_link" in kwargs and "subitem_link" not in docs[i].metadata:
            docs[i].metadata["subitem_link"] = kwargs["subitem_link"]
        if "title" not in docs[i].metadata or docs[i].metadata["title"] == "Untitled":
            if "title" in kwargs and kwargs["title"] and kwargs["title"] != "Untitled":
                docs[i].metadata["title"] = kwargs["title"]
            elif "http" in docs[i].metadata["path"].lower():
                docs[i].metadata["title"] = get_url_title(docs[i].metadata["path"])
                if not docs[i].metadata["title"]:
                    docs[i].metadata["title"] = "Untitled"
                    red(f"Could not get title from {path}")
        if "title" in kwargs and kwargs["title"] != docs[i].metadata["title"] and kwargs["title"] not in docs[i].metadata["title"]:
            docs[i].metadata["title"] += " - " + kwargs["title"]
        if "playlist_title" in kwargs:
            docs[i].metadata["title"] = kwargs["playlist_title"] + " - " + docs[i].metadata["title"]

        if "docs_reading_time" not in docs[i].metadata:
            if not total_reading_length:
                total_reading_length = sum([len(d.page_content) for d in docs]) / average_word_length / wpm
                assert total_reading_length > 0.5, f"Failing doc: total reading length is suspiciously low for {docs[i].metadata}"
            docs[i].metadata["docs_reading_time"] = total_reading_length
        if "source" not in docs[i].metadata:
            if "path" in docs[i].metadata:
                docs[i].metadata["source"] = docs[i].metadata["path"]
            else:
                docs[i].metadata["source"] = docs[i].metadata["title"]

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
        embeddings = RollingWindowEmbeddings(
                model_name=embed_model,
                encode_kwargs={
                    # "batch_size": 1,
                    "show_progress_bar": True,
                    "normalize_embeddings": True,
                    },
                )
        if "stopwords" in kwargs:
            embed_args["stopwords"] = kwargs["stopwords"]

    lfs = LocalFileStore(f".cache/embeddings/{embed_model}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            lfs,
            namespace=embed_model,
            )

    # reload passed embeddings
    if loadfrom:
        red("Reloading documents and embeddings from file")
        path = Path(loadfrom)
        assert path.exists(), f"file not found at '{path}'"
        db = FAISS.load_local(str(path), cached_embeddings)
        return db

    red("\nLoading embeddings.")

    docs = loaded_docs
    if len(docs) >= 50:
        docs = sorted(docs, key=lambda x: random.random())

    # check price of embedding
    full_tkn = sum([get_tkn_length(doc.page_content) for doc in docs])
    red(f"Total number of tokens in documents (not checking if already present in cache): '{full_tkn}'")
    if embed_model == "openai":
        dol_price = full_tkn * 0.0001 / 1000
        red(f"With OpenAI embeddings, the total cost for all tokens is ${dol_price:.4f}")
        if dol_price > 1:
            ans = input(f"Do you confirm you are okay to pay this? (y/n)\n>")
            if ans.lower() not in ["y", "yes"]:
                red("Quitting.")
                raise SystemExit()

    def get_embedding_stripping_stopwords(doc, embeddings, embed_args=embed_args):
        whi("Computing embeddings")
        prev = doc.page_content
        doc.page_content = re.sub(charac_regex, " ", doc.page_content.lower())
        for reg in embed_args["stopwords"]:
            doc.page_content = re.sub(reg, " ", doc.page_content)
        temp = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        for k in temp.docstore.__dict__.keys():
            for kk in temp.docstore.__dict__[k].keys():
                temp.docstore.__dict__[k][kk].page_content = prev
        return temp, doc.metadata['path']

    if any("stopwords" in d.metadata for d in docs):
        whi("Using diy embedding function that strips stopwords")
        results = Parallel(
                n_jobs=1 if not debug else 3,
                backend="threading",
                )(delayed(get_embedding_stripping_stopwords
                          )(
                              doc=doc,
                              embeddings=cached_embeddings
                              ) for doc in tqdm(
                                  docs,
                                  desc="embedding documents"))
        # merge the results
        for i, temp, path in enumerate(results):
            if not i:
                db = temp
            else:
                db.merge_from(temp)
    else:
        t = time.time()
        whi(f"Creating FAISS index for {len(docs)} documents")
        db = FAISS.from_documents(
                docs,
                cached_embeddings,
                normalize_L2=True,
                )
        whi(f"Done creating index in {time.time()-t:.2f}s")

    # saving embeddings
    if saveas:
        path = Path(saveas)
        db.save_local(str(path))

    return db, cached_embeddings

@loaddoc_cache.cache
def load_youtube_playlist(playlist_url):
    with youtube_dl.YoutubeDL({"quiet": False}) as ydl:
        try:
            loaded = ydl.extract_info(playlist_url, download=False)
        except (KeyError, DownloadError, ExtractorError) as e:
            raise Exception(red(f"ERROR: Youtube playlist link skipped because : error during information \
        extraction from {playlist_url} : {e}"))
    return loaded

@loaddoc_cache.cache(ignore=["loader"])
def cached_yt_loader(loader, path, add_video_info, language, translation):
    yel(f"Not using cache for youtube {path}")
    docs = loader(
            path,
            add_video_info=add_video_info,
            language=language,
            translation=translation,
            ).load()
    return docs

def create_hyde_retriever(
        query,
        filetype,

        llm,
        top_k,

        embed_model,
        embeddings,
        embeddings_engine,

        kwargs,
        loadfrom,
        debug
        ):
    """
    create a retriever only for the subset of documents from the
    loaded_embeddings that were found using HyDE technique (i.e. asking
    the llm to create a hypothetical answer and use the embedding of this
    answer to search similar content)

    The code is a little strange because it actually reloads only a portion
    of the embeddings from cache if possible.

    https://python.langchain.com/docs/use_cases/question_answering/how_to/hyde
    """

    HyDE_template = """Please imagine the answer to the user's question about a document:
    Document type: [[filetype]]
    User question: {question}
    Answer:""".replace("[[filetype]]", filetype)
    hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=HyDE_template,
            )

    hyde_chain = LLMChain(
            llm=llm,
            prompt=hyde_prompt,
            )

    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=hyde_chain,
        base_embeddings=embeddings_engine,
        )
    hyde_vector = hyde_embeddings.embed_query(query)

    hyde_doc = embeddings.similarity_search_by_vector(
            embedding=hyde_vector,
            k=top_k,
            )
    vecstore, _ = load_embeddings(
            embed_model=embed_model,
            loadfrom = loadfrom,
            saveas=None,
            debug=debug,
            loaded_docs=hyde_doc,
            kwargs=kwargs,
            )

    retriever = vecstore.as_retriever(
        search_kwargs={"k": top_k, "distance_metric": "cos"}
        )

    return retriever


def create_parent_retriever(
        task,
        loaded_embeddings,
        loaded_docs,
        ):
    "https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever"
    # parent retriever
    csp = get_splitter(task)
    psp = get_splitter(task)
    psp._chunk_size *= 4
    parent = ParentDocumentRetriever(
            vectorstore=loaded_embeddings,
            docstore=LocalFileStore(".cache/parent_retriever"),
            child_splitter=csp,
            parent_splitter=psp,
            )
    parent.add_documents(loaded_docs)
    return parent
