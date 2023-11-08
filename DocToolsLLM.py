from joblib import Parallel, delayed
from threading import Lock
from pathlib import Path
import time
from datetime import datetime
import re
import textwrap
import fire
import os
from tqdm import tqdm
import signal
import pdb
from ftlangdetect import detect as language_detect

from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.docstore.document import Document
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever, KNNRetriever, SVMRetriever
from langchain.prompts.prompt import PromptTemplate

from utils.llm import load_llm, AnswerConversationBufferMemory
from utils.file_loader import load_doc, load_embeddings, create_hyde_retriever, get_tkn_length, average_word_length, wpm, get_splitter, check_docs_tkn_length, create_parent_retriever
from utils.logger import whi, yel, red
from utils.cli import ask_user
from utils.tasks import do_summarize

os.environ["TOKENIZERS_PARALLELISM"] = "true"

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

class DocToolsLLM:
    VERSION = 0.9
    def __init__(
            self,
            model="openai",
            task="query",
            filetype="infer",
            local_llm_path=None,
            # embed_model="openai",
            embed_model = "paraphrase-multilingual-mpnet-base-v2",
            # embed_model = "distiluse-base-multilingual-cased-v1",
            # embed_model = "msmarco-distilbert-cos-v5",
            # embed_model = "all-mpnet-base-v2",
            saveas=".cache/latest_docs_and_embeddings",
            loadfrom=None,

            top_k=10,
            n_recursive_summary=3,
            n_summaries_target=-1,

            debug=False,
            llm_verbosity=True,

            help=False,
            h=False,
            **kwargs,
            ):
        """
        Parameters
        ----------
        --task str, default query
            possibilities:
                * query means to load the input files then wait for user question.
                * summarize means the input will be passed through a summarization prompt.
                * summarize_then_query
                * summarize_link_file takes in --filetype must be link_file

        --filetype str, default infer
            the type of input. Depending on the value, different other parameters
            are needed. If json_list is used, the line of the input file can contain
            any of those parameters as long as they are as json. You can find
            an example of json_list file in utils/json_list_example.txt

            Supported values => relevant parameters
                * youtube => --path must be a link to youtube --language=["fr","en"] to use french transcripts if possible, english otherwise --translation=en to use the transcripts after translation to english
                * youtube_playlist => --path must link to a youtube playlist. language and translation are set to their default value of fr,en and en
                * pdf => --path is path to pdf
                * txt => --path is path to txt
                * url => --path must be a valid http(s) link
                * anki => --anki_profile is the name of the profile --anki_deck the beginning of the deckname --anki_notetype the beginning of the notetype to keep --anki_fields list of fields to keep
                * string => no other parameters needed, will ask to provide a string
                * local_audio => needs whisper_prompt and whisper_lang

                * json_list => --path is path to a txt file that contains a json for each line containing at least a filetype and a path key/value but can contain any parameters described here
                * recursive => --path is the starting path --pattern is the globbing patterns to append --exclude and --include can be a list of regex applying to found paths (include is run first then exclude, if the pattern is only lowercase it will be case insensitive) --recursed_filetype is the filetype to use for each of the found path
                * link_file => --path must point to a file where each line is a link that will be summarized. The resulting summary will be added to --out_file. Links that have already been summarized in out_file will be skipped (the out_file is never overwritten). If a line is a markdown linke like [this](link) then it will be parsed as a link. Empty lines and starting with # are ignored. If argument --out_file_logseq_mode is present, the formatting will be compatible with logseq.

                * "infer" => can often be used in the backend to try to guess the proper filetype. Experimental.

        --model str, default openai
            either gpt4all, llama, openai or fake/test/testing to use a fake answer.

        --local_llm_path str
            if model is not openai, this needs to point to a compatible model

        --embed_model str, default "openai"
            Either 'openai' or sentence_transformer embedding model to use.
            If you change this, the embedding cache will be usually
            need to be recomputed with new elements (the hash
            used to check for previous values includes the name of the model
            name)

        --saveas str, default .cache/latest_docs_and_embeddings
            only used if task is query
            save the latest 'inputs' to a file. Can be loaded again with
            --loadfrom to speed up loading time. This loads both the
            split documents and embeddings but will not update itself if the
            original files have changed.

        --loadfrom str, default None
            path to the file saved using --saveas

        --top_k int, default 10
            number of chunks to look for when querying

        --n_recursive_summary int, default 2
            will always recursively summarize

        --n_summaries_target int, default -1
            Only active if query is 'summarize_link_file'. Set a limit to
            the number of links that will be summarized. If the number of
            TODO in the output is higher, exit. If it's lower, only do the
            difference. -1 to disable.

        --debug bool, default False
            if True will open a debugger instead before crashing, also use
            sequential processing instead of multithreading and enable
            langchain tracing.

        --llm_verbosity, default True
            if True, will print the intermediate reasonning steps of LLMs

        --help or -h, default False
            if True, will return this documentation.
        """
        if help or h:
            print(self.__init__.__doc__)
            return

        # checking argument validity
        assert "loaded_docs" not in kwargs, "'loaded_docs' cannot be an argument as it is used internally"
        assert "loaded_embeddings" not in kwargs, "'loaded_embeddings' cannot be an argument as it is used internally"
        assert task in ["query", "summarize", "summarize_then_query", "summarize_link_file"], "invalid task value"
        assert isinstance(filetype, str), "filetype must be a string"
        if task in ["summarize", "summarize_then_query"]:
            assert not loadfrom, "can't use loadfrom if task is summary"
        assert (task == "summarize_link_file" and filetype == "link_file"
                ) or (task != "summarize_link_file" and filetype != "link_file"
                        ), "summarize_link_file must be used with filetype link_file"
        if task == "summarize_link_file":
            assert "path" in kwargs, 'missing path arg for summarize_link_file'
            assert "out_file" in kwargs, 'missing "out_file" arg for summarize_link_file'
            assert kwargs["out_file"] != kwargs["path"], "can't use same 'path' and 'out_file' arg"
        assert "/" not in embed_model, "embed model can't contain slash"
        assert isinstance(n_summaries_target, int), "invalid type of n_summaries_target"

        for k in kwargs:
            if k not in [
                    "anki_profile", "anki_notetype", "anki_fields", "anki_deck",
                    "whisper_lang", "whisper_prompt",
                    "path", "include", "exclude",
                    "out_file", "out_file_logseq_mode",
                    "language", "translation",
                    "out_check_file",
                    ]:
                red(f"Found unexpected keyword argument: '{k}'")

        if filetype == "string":
            top_k = 1
            red("Input is 'string' so setting 'top_k' to 1")

        # storing as attributes
        self.model = model
        self.task = task
        self.filetype = filetype
        self.local_llm_path = local_llm_path
        self.embed_model = embed_model
        self.saveas = saveas
        self.loadfrom = loadfrom
        self.top_k = top_k
        self.debug = debug
        self.kwargs = kwargs
        self.llm_verbosity = llm_verbosity
        self.n_recursive_summary = n_recursive_summary
        self.n_summaries_target = n_summaries_target

        if self.debug:
            # make the script interruptible
            signal.signal(signal.SIGINT, (lambda signal, frame : pdb.set_trace()))
            os.environ["LANGCHAIN_TRACING"] = "true"

        # compile include / exclude regex
        if "include" in self.kwargs:
            for i, inc in enumerate(self.kwargs["include"]):
                if inc == inc.lower():
                    self.kwargs["include"][i] = re.compile(inc, flags=re.IGNORECASE)
                else:
                    self.kwargs["include"][i] = re.compile(inc)
        if "exclude" in self.kwargs:
            for i, exc in enumerate(self.kwargs["exclude"]):
                if exc == exc.lower():
                    self.kwargs["exclude"][i] = re.compile(exc, flags=re.IGNORECASE)
                else:
                    self.kwargs["exclude"][i] = re.compile(exc)

        # loading llm
        self.llm, self.callback = load_llm(model, local_llm_path)

        # loading documents
        if not loadfrom:
            self.loaded_docs = load_doc(
                    filetype=self.filetype,
                    debug=self.debug,
                    task=self.task,
                    **self.kwargs)
            if len(self.loaded_docs) > 1:
                assert id(self.loaded_docs[0].metadata) != id(self.loaded_docs[-1].metadata), (
                        "Same metadata object is used to store information on "
                        "multiple documents!")
            hashes = [d.metadata["hash"] for d in self.loaded_docs]
            if len(set(hashes)) != len(hashes):
                red("Found duplicate hashes after loading documents:")
                for i, doc in enumerate(self.loaded_docs):
                    n = hashes.count(doc.metadata["hash"])
                    while n > 1:
                        red(f"  * Removed #{i}: {doc}")
                        self.loaded_docs[i] = None
                        n -= 1
                self.loaded_docs = [d for d in self.loaded_docs if d is not None]

        else:
            self.loaded_docs = None  # will be loaded when embeddings are loaded

        _ = self.process_task()

        whi("Done with tasks.")
        if self.debug:
            breakpoint()


    def process_task(self):
        red(f"\nProcessing task '{self.task}'")

        if self.task in ["summarize_link_file", "summarize", "summarize_then_query"]:
            # storing links in dict instead of set to keep the original ordering
            links_todo = {}
            already_done = {}
            failed = []

            # get the list of documents from the same source. Also checks if
            # it's not part of the output file if task is "summarize_link_file"
            if self.task == "summarize_link_file":
                if not Path(self.kwargs["out_file"]).exists():
                    Path(self.kwargs["out_file"]).touch()
                with open(self.kwargs["out_file"], "r") as f:
                    output_content = f.read()

                if "out_check_file" in self.kwargs:
                    # this is an undocumented function for the author. It
                    # allows to specify a second path for which to check if
                    # a document has already been summaried. I use this because
                    # I made a script to automatically move my DONE tasks
                    # from logseq to another near by file.
                    assert Path(self.kwargs["out_check_file"]).exists()
                    with open(self.kwargs["out_check_file"], "r") as f:
                        output_content += f.read()

                for d in self.loaded_docs:
                    assert "subitem_link" in d.metadata, "missing 'subitem_link' in a doc metadata"

                    link = d.metadata["subitem_link"]
                    if link in already_done or link in links_todo:
                        continue
                    if link in output_content:
                        whi(f"Skipping link : already summarized in out_file: '{link}'")
                        already_done[link] = None
                        continue

                    if len(links_todo) < self.n_summaries_target:
                        links_todo[link] = None
                    else:
                        yel("'n_summaries_target' limit reached, will not add more links to summarize for this run.")
                        break

                # comment out the links that are marked as already done
                # if already_done:
                #     with open(self.kwargs["path"], "r") as f:
                #         temp = f.read().split("\n")
                #     with open(self.kwargs["path"], "w") as f:
                #         for t in temp:
                #             for done_link in already_done:
                #                 if done_link in t:
                #                     t = f"#already done as of {today}# {t}"
                #                     break
                #             f.write(t.strip() + "\n")

                if self.n_summaries_target > 0:
                    # allows to run DocTools to summarise from a link file
                    # only if there are less than 'n_summaries_target' TODOS
                    # blocks in the target file. This way we can have a
                    # list of TODOS that will never be larger than this.
                    # Avoiding both having too many summaries and not enough
                    # as it allows to run this frequently
                    n_todos_desired = self.n_summaries_target
                    assert isinstance(n_todos_desired, int)
                    n_todos_present = output_content.count("- TODO ")
                    if n_todos_present >= n_todos_desired:
                        return red(f"Found {n_todos_present} in the output file(s) which is >= {n_todos_desired}. Exiting without summarising.")
                    else:
                        self.n_summaries_target = n_todos_desired - n_todos_present
                        red(f"Found {n_todos_present} in output file(s) which is under {n_todos_desired}. Will summarize only {self.n_summaries_target}")
                        assert self.n_summaries_target > 0

                    while len(links_todo) > self.n_summaries_target:
                        del links_todo[list(links_todo.keys())[-1]]

                # estimate price before summarizing, in case you put the bible in there
                docs_tkn_cost = {}
                for doc in self.loaded_docs:
                    meta = doc.metadata["subitem_link"]
                    if meta in links_todo:
                        if meta not in docs_tkn_cost:
                            docs_tkn_cost[meta] = get_tkn_length(doc.page_content)
                        else:
                            docs_tkn_cost[meta] += get_tkn_length(doc.page_content)

            else:
                for d in self.loaded_docs:
                    links_todo[d.metadata["path"]] = None
                assert len(links_todo) == 1, f"Invalid length of links_todo for this task: '{len(links_todo)}'"

                docs_tkn_cost = {}
                for doc in self.loaded_docs:
                    meta = doc.metadata["path"]
                    if meta not in docs_tkn_cost:
                        docs_tkn_cost[meta] = get_tkn_length(doc.page_content)
                    else:
                        docs_tkn_cost[meta] += get_tkn_length(doc.page_content)

            full_tkn = sum(list(docs_tkn_cost.values()))
            red("Token price of each document:")
            for k, v in docs_tkn_cost.items():
                red(f"- {v:>6}: {k}")

            red(f"Total number of tokens in documents to summarize: '{full_tkn}'")
            # a conservative estimate is that it takes 4 times the number
            # of tokens of a document to summarize it
            estimate_tkn = 2.4 * full_tkn
            if self.n_recursive_summary > 0:
                estimate_tkn += sum([full_tkn / ((i + 1) * 4) for i, ii in enumerate(range(self.n_recursive_summary))])
            estimate_dol = estimate_tkn / 1000 * 0.0016
            red(f"Conservative estimate of the cost to summarize: ${estimate_dol:.4f} for {estimate_tkn} tokens.")
            if estimate_dol > 1:
                raise Exception(red("Cost estimate > $1 which is absurdly high. Has something gone wrong? Quitting."))

            if self.model == "openai":
                # increase likelyhood that chatgpt will use indentation by
                # biasing towards adding space.
                self.llm.model_kwargs["logit_bias"] = {
                        12: 4,  # '-'
                        # 220: 1,  # ' '
                        # 532: 1,  # ' -'
                        #9: 10,  # '*'
                        #1635: 10,  # ' *'
                        197: 4,  # '\t'
                        334: 4,  # '**'
                        }
                self.llm.model_kwargs["frequency_penalty"] = 0.5
                self.llm.model_kwargs["temperature"] = 0.0

            def threaded_summary(link, lock):
                if self.task == "summarize_link_file":
                    # get only the docs that match the link
                    relevant_docs = [d for d in self.loaded_docs if d.metadata["subitem_link"] == link]
                else:
                    relevant_docs = self.loaded_docs
                assert relevant_docs, 'Empty relevant_docs!'

                # parse metadata from the doc
                metadata = []
                if "title" in relevant_docs[0].metadata:
                    item_name = f"{relevant_docs[0].metadata['title'].strip()} - {link}"
                    metadata.append(f"Title: '{item_name.strip()}'")
                else:
                    item_name = link
                if "docs_reading_time" in relevant_docs[0].metadata:
                    doc_reading_length = relevant_docs[0].metadata["docs_reading_time"]
                    metadata.append(f"Reading length: {doc_reading_length:.1f} minutes")
                else:
                    doc_reading_length = None
                if "author" in relevant_docs[0].metadata:
                    author = relevant_docs[0].metadata["author"].strip()
                    metadata.append(f"Author: '{author}'")
                else:
                    author = None

                # detect language
                lang_info = language_detect(relevant_docs[0].page_content.replace("\n", "<br>"))
                if lang_info["score"] >= 0.8:
                    lang = lang_info['lang']
                    if lang == "fr":
                        lang = "FRENCH"
                    else:  # prefer english to anything other than french
                        lang = "ENGLISH"
                else:
                    lang = "ENGLISH"
                    red(f"Language detection failed: '{lang_info}'")

                if metadata:
                    metadata = "- Text metadata:\n\t- " + "\n\t- ".join(metadata) + "\n"
                    metadata += "\t- Section number: [PROGRESS]\n"
                else:
                    metadata = ""

                # summarize each chunk of the link and return one text
                summary, n_chunk, doc_total_tokens, doc_total_cost = do_summarize(
                        docs=relevant_docs,
                        metadata=metadata,
                        language=lang,
                        model=self.model,
                        llm=self.llm,
                        callback=self.callback,
                        verbose=self.llm_verbosity,
                        )

                # get reading length of the summary
                sum_reading_length = len(summary) / average_word_length / wpm
                whi(f"{item_name} reading length is {sum_reading_length:.1f}")

                n_recursion_done = 0
                if self.n_recursive_summary > 0:
                    splitter = get_splitter(self.task)
                    summary_text = summary
                    if metadata:
                        metadata = metadata.strip() + "\n\t- New task: enhance this summary while respecting the rules\n"
                    for n_recur in range(self.n_recursive_summary):
                        red(f"Doing recursive summary #{n_recur} of {item_name}")

                        # remove any chunk count that is not needed to summarize
                        sp = summary_text.split("\n")
                        for i, l in enumerate(sp):
                            if l.strip() == "- ---":
                                sp[i] = None
                            elif re.search(r"- Chunk \d+/\d+", l):
                                sp[i] = None
                        summary_text = "\n".join([s.rstrip() for s in sp if s])

                        summary_docs = [Document(page_content=summary_text)]
                        summary_docs = splitter.transform_documents(summary_docs)
                        try:
                            check_docs_tkn_length(summary_docs, item_name)
                        except Exception as err:
                            red(f"Exception when checking if {item_name} could be recursively summarized for the #{n_recur} time: {err}")
                            break
                        summary_text, n_chunk, new_doc_total_tokens, new_doc_total_cost = do_summarize(
                                docs=summary_docs,
                                metadata=metadata,
                                language=lang,
                                model=self.model,
                                llm=self.llm,
                                callback=self.callback,
                                verbose=self.llm_verbosity,
                                )
                        doc_total_tokens += new_doc_total_tokens
                        doc_total_cost += new_doc_total_cost
                        n_recursion_done += 1
                        sum_reading_length = len(summary_text) / average_word_length / wpm
                        whi(f"{item_name} reading length after recursion #{n_recur} is {sum_reading_length:.1f}")
                    summary = summary_text



                # make sure to use the same markdown formatting
                summary = summary.replace("* ", "- ")
                summary = summary.replace("- - ", "- ")

                with lock:
                    red(f"\n\nSummary of '{link}':\n{summary}")

                    red(f"Tokens used for {link}: '{doc_total_tokens}' (${doc_total_cost:.5f})")

                if "out_file_logseq_mode" in self.kwargs:
                    header = f"\n- TODO {item_name}"
                    header += "\n\tcollapsed:: true"
                    header += "\n\tblock_type:: DocToolsLLM_summary"
                    header += f"\n\tDocToolsLLM_version:: {self.VERSION}"
                    header += f"\n\tDocToolsLLM_model:: {self.model}"
                    header += f"\n\tDocToolsLLM_parameters:: n_recursion_summary={self.n_recursive_summary};n_recursion_done={n_recursion_done}"
                    header += f"\n\tsummary_date:: {today}"
                    header += f"\n\tsummary_timestamp:: {int(time.time())}"
                    header += f"\n\ttoken_cost:: {doc_total_tokens}"
                    header += f"\n\tdollar_cost:: {doc_total_cost:.5f}"
                    header += f"\n\tsummary_reading_time:: {sum_reading_length:.1f}"
                    if doc_reading_length:
                        header += f"\n\tdoc_reading_time:: {doc_reading_length:.1f}"
                        header += f"\n\treading_time_prct_speedup:: {int(sum_reading_length/doc_reading_length * 100)}%"
                    if n_chunk > 1:
                        header += f"\n\tchunks:: {n_chunk}"
                    if author:
                        header += f"\n\tauthor:: {author}"
                    if lang:
                        header += f"\n\tlanguage:: {lang}"

                else:
                    header = f"\n- {item_name}    cost: {doc_total_tokens} (${doc_total_cost:.5f})"
                    if doc_reading_length:
                        header += f"    {doc_reading_length:.1f} minutes"
                    if author:
                        header += f"    by '{author}'"
                    header += f"    DocToolsLLM version {self.VERSION} with model {self.model}"
                    header += f"    parameters: n_recursion_summary={self.n_recursive_summary};n_recursion_done={n_recursion_done}"

                # save to output file
                if "out_file" in self.kwargs:
                    with lock:
                        with open(self.kwargs["out_file"], "a") as f:
                            f.write(header)
                            for bulletpoint in summary.split("\n"):
                                f.write("\n")
                                # make sure the line begins with a bullet point
                                if not bulletpoint.strip().startswith("- "):
                                    begin_space = re.search(r"^(\s+)", bulletpoint)
                                    if not begin_space:
                                        begin_space = [""]
                                    bulletpoint = begin_space[0] + "- " + bulletpoint
                                f.write(f"    {bulletpoint}")
                            f.write("\n\n\n")
                return {
                        "link": link,
                        "sum_reading_length": sum_reading_length,
                        "doc_reading_length": doc_reading_length,
                        "doc_total_tokens": doc_total_tokens,
                        "doc_total_cost": doc_total_cost,
                        "summary": summary,
                        }

            lock = Lock()
            results = Parallel(
                    n_jobs=3 if not self.debug else 1,
                    backend="threading",
                    )(delayed(threaded_summary)(
                        link=link,
                        lock=lock,
                        ) for link in tqdm(
                            links_todo,
                            desc="Summarizing documents",
                            disable=(not len(links_todo) - 1) or self.debug,
                            ))
            total_tkn_cost = sum([x["doc_total_tokens"] for x in results])
            total_dol_cost = sum([x["doc_total_cost"] for x in results])
            total_docs_length = sum([x["doc_reading_length"] for x in results])
            total_summary_length = sum([x["sum_reading_length"] for x in results])

            red(f"Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})")
            red(f"Total time saved by this run: {total_docs_length:.1f} minutes")

            # if "out_file" in self.kwargs:
            #     # after summarizing all links, append to output file the total cost
            #     if total_tkn_cost != 0 and total_dol_cost != 0:
            #         with open(self.kwargs["out_file"], "a") as f:
            #             f.write(f"- Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
            #             f.write(f"- Total time saved by this run: {total_docs_length - total_summary_length:.1f} minutes\n\n\n")

            # and write to input file a summary too
            if "out_file" in self.kwargs:
                try:
                    with open(self.kwargs["path"], "a") as f:
                        f.write(f"\n\n")
                        f.write(f"- Done with summaries of {today}\n")
                        f.write(f"    - Number of links summarized: {len(links_todo) - len(failed)}/{len(links_todo) + len(already_done)}\n")
                        if failed:
                            f.write(f"    - Number of links failed: {len(failed)}:\n")
                            for f in failed:
                                f.write(f"        - {f}\n")
                        # f.write(f"    - Total cost of this run: '{total_tkn_cost}' (${total_dol_cost:.5f})\n")
                        # f.write(f"    - Total time saved by this run: plausibly {total_docs_length:.1f} minutes\n")
                except Exception as err:
                    red(f"Exception when writing end of run details to input file: '{err}'")

            if self.task == "summary_then_query":
                whi("Done summarizing. Switching to query mode.")
                if self.model == "openai":
                    del self.llm.model_kwargs["logit_bias"]
            else:
                whi("Done summarizing. Exiting.")
                raise SystemExit()

        # load embeddings for querying
        self.loaded_embeddings, self.embeddings = load_embeddings(
                self.embed_model,
                self.loadfrom,
                self.saveas,
                self.debug,
                self.loaded_docs,
                self.kwargs)

        assert self.task in ["query", "summarize_then_query"]

        # set default ask_user argument
        multiline = False

        # conversational memory
        memory = AnswerConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True)

        cli_commands = {
                "top_k": self.top_k,
                "multiline": multiline,
                "retriever": "all",
                "task": self.task,
                }
        while True:
            try:
                with self.callback() as cb:
                    query, cli_commands = ask_user(
                            "\n\nWhat is your question? (Q to quit)\n",
                            cli_commands,
                            )

                    retrievers = []
                    if cli_commands["retriever"] in ["hyde", "all"]:
                        retrievers.append(
                                create_hyde_retriever(
                                    query=query,
                                    filetype=self.filetype,

                                    llm=self.llm,
                                    top_k=cli_commands["top_k"],

                                    embed_model=self.embed_model,
                                    embeddings=self.loaded_embeddings,
                                    embeddings_engine=self.embeddings,

                                    loadfrom=self.loadfrom,
                                    kwargs=self.kwargs,
                                    debug=self.debug,
                                    )
                                )

                        # retrievers.append(
                        #         KNNRetriever.from_texts(
                        #             [d.page_content for d in self.loaded_docs],
                        #             self.embeddings,
                        #             )
                        #         )
                        # retrievers.append(
                        #         SVMRetriever.from_texts(
                        #             [d.page_content for d in self.loaded_docs],
                        #             self.embeddings,
                        #             )
                        #         )

                        retrievers.append(
                                create_parent_retriever(
                                    task=self.task,
                                    loaded_embeddings=self.loader_embeddings,
                                    loaded_docs=self.loaded_docs,
                                    )
                                )

                    if cli_commands["retriever"] in ["simple", "all"]:
                        retrievers.append(
                                self.loaded_embeddings.as_retriever(
                                    search_kwargs={
                                        "k": cli_commands["top_k"],
                                        "distance_metric": "cos",
                                        })
                                    )

                    if len(retrievers) == 1:
                        retriever = retrievers[0]
                    else:
                        whi("Merging multiple retrievers")
                        retriever = MergerRetriever(retrievers)

                        # remove redundant results from the merged retrievers:
                        filtered = EmbeddingsRedundantFilter(embeddings=self.embeddings)
                        pipeline = DocumentCompressorPipeline(transformers=[filtered])
                        retriever = ContextualCompressionRetriever(
                            base_compressor=pipeline, base_retriever=retriever
                        )

                    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

                    Chat History:
                    {chat_history}

                    Follow Up Input: {question}

                    Standalone question:"""
                    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
                    question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
                    doc_chain = load_qa_with_sources_chain(self.llm, chain_type="map_reduce")

                    chain = ConversationalRetrievalChain(
                            retriever=retriever,
                            question_generator=question_generator,
                            combine_docs_chain=doc_chain,
                            return_source_documents=True,
                            return_generated_question=True,
                            verbose=self.llm_verbosity,
                            memory=memory,
                            )


                    ans = chain(
                            inputs={
                                "question": query,
                                },
                            return_only_outputs=False,
                            include_run_info=True,
                            )

                    # docs = self.loaded_embeddings.similarity_search(
                    #         query,
                    #         k=cli_commands["top_k"],
                    #         )
                    # chain = load_qa_with_sources_chain(
                    #         llm=self.llm,
                    #         #retriever=retriever,
                    #         chain_type="map_reduce",
                    #         #prompt=query_prompt,
                    #         combine_prompt=combine_prompt,
                    #         #return_map_steps=True,
                    #         #return_source_documents=True,
                    #         verbose=self.llm_verbosity,
                    #         input_key="question",
                    #         )

                    # ans = chain(
                    #         inputs={
                    #             "question": query,
                    #             "input_documents": docs,
                    #             },
                    #         return_only_outputs=False,
                    #         include_run_info=True,
                    #         )

                whi("\n\nSources:")
                for doc in ans["source_documents"]:
                    keys = doc.metadata.keys()
                    for toprint in [
                            "filetype", "path", "nid", "anki_deck", "anki_tags"]:
                        if toprint in keys:
                            val = doc.metadata[toprint]
                            yel(f"    * {toprint}: {val}")

                    toignore = [k for k in keys if k not in toprint]
                    whi(f"Metadata not printed: '{','.join(toignore)}'")

                    content = doc.page_content.strip()
                    wrapped = "\n".join(textwrap.wrap(content, width=120))
                    whi("    * content:")
                    whi(f"{wrapped:>10}")
                    print("\n\n")

                red(f"Answer:\n{ans['answer']}\n")

                yel(f"Tokens used: '{cb.total_tokens}' (${cb.total_cost:.5f})")

            except Exception as err:
                whi(f"Error: '{err}'")
                raise


if __name__ == "__main__":
    instance = fire.Fire(DocToolsLLM)
